#!/usr/bin/env python3
"""
儿童涂色卡生成器 v2（推荐方案一：单输入 idea）
- OpenRouter chat/completions
- system / user 分离 messages（更稳）
- 支持：文生图（idea->image）、图生图（image->coloring）
- 去掉：PNG->PDF / 矢量化 / reportlab / cairosvg / potrace 等后处理
- 输出：PNG + JSON 记录
- 说明：OpenRouter 对 Gemini 系列可用 image_config 控制比例/分辨率；
        OpenAI 图像模型经由 OpenRouter chat/completions 可能忽略 size，回落到 1024x1024。
- 如需稳定的竖版/横版尺寸控制，建议使用 --api openai（OpenAI 官方 Images API）。

环境变量：
  OPENROUTER_API_KEY=xxxx

用法示例：
  python coloring_generator.py --batch --model gpt5-image-mini --out output_batch
  python coloring_generator.py --generate "A friendly dragon flying over a castle" --kids --portrait --model nano-banana-pro
  python coloring_generator.py --image2color ./input.jpg --kids --portrait --model gpt5-image-mini
  python coloring_generator.py --interactive
"""

import os
import json
import time
import base64
import argparse
import requests
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

# -------------------------
# .env 加载（可选）
# -------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# 基础枚举与输入结构
# ============================================================

class AgeMode(Enum):
    TODDLER = "toddler"
    KIDS = "kids"


class Orientation(Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    AUTO = "auto"


class ApiBackend(Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"


@dataclass
class IdeaInput:
    """单输入 idea 模式"""
    idea: str
    age_mode: AgeMode = AgeMode.KIDS
    orientation: Orientation = Orientation.AUTO

# ============================================================
# Prompt System（融合版，生产级）
# ============================================================

class ColoringPromptSystem:
    """
    Prompt 架构：
      - system_prompt 固定：质量/印刷/安全底线
      - user_prompt 动态：idea + age + orientation
    """

    SYSTEM_PROMPT = """You are a professional children's coloring book illustrator
specialized in creating print-ready black-and-white coloring pages
that children love to color.

Your mission is to combine imagination, beauty, and clarity,
while ALWAYS keeping the artwork easy to color and safe to print.

PRIORITY ORDER:
1. Child usability and enjoyment
2. Print safety and clarity
3. Visual appeal and imagination

MANDATORY LINE & PRINT RULES:
- Pure black outlines on a pure white background only
- Clean, smooth, continuous lines
- All shapes and regions must be fully closed
- Uniform line thickness throughout the entire illustration
- No filled black areas, no shading, no gradients, no textures
- No overlapping, tangled, or doubled lines
- No tiny gaps, thin corridors, or cramped spaces
- No text, letters, numbers, logos, or symbols

DESIGN & COLORING USABILITY:
- All coloring areas must be large and comfortable to fill
- Prefer rounded, friendly shapes over sharp or narrow ones
- Maintain clear spacing between separate objects
- Every enclosed region must be safely colorable on its own

ARTISTIC & IMAGINATION DIRECTION:
- Cheerful, playful, and storybook-like mood
- Friendly characters with simple, expressive faces
- Whimsical and magical feeling created through composition,
  not through excessive detail
- Scenes should invite curiosity, joy, and gentle imagination
- Visual richness must remain clean, open, and uncluttered

COMPOSITION & LAYOUT:
- Balanced, harmonious page composition
- Clear white margins suitable for printing
- Illustration should naturally fill the page without empty corners
- Overall style should resemble high-quality professional children's coloring books

FINAL SELF-CHECK BEFORE OUTPUT:
- Every outline is closed and continuous
- No dark or filled regions appear
- Line weight is consistent and print-friendly
- Complexity matches the target age group
- The page is ready for direct printing and enjoyable coloring
"""

    AGE_MODE_PROMPTS = {
        AgeMode.TODDLER: """[AGE MODE: TODDLER / PRESCHOOL]
Target complexity: VERY LOW
- 3–5 friendly objects maximum
- Extra-large, open coloring areas
- Thick, bold outlines
- Wide spacing between all elements
- Instantly recognizable subjects

Imagination style:
Soft, comforting, and friendly.
Think smiling suns, fluffy clouds,
cute animals with big eyes and simple shapes.
""",
        AgeMode.KIDS: """[AGE MODE: KIDS]
Target complexity: MODERATE
- 5–10 clear objects
- Medium to large coloring areas
- Simple decorative elements allowed
- Light background scenery for storytelling
- Characters may show action and emotion

Imagination style:
Playful and adventurous.
Think secret gardens, tree houses,
underwater worlds, friendly dragons,
and gentle magical scenes.
Decorative details must remain large and simple.
"""
    }

    ORIENTATION_PROMPTS = {
        Orientation.PORTRAIT: """VERTICAL COMPOSITION:
- Tall, storybook-style layout
- Sky elements near the top
- Ground elements near the bottom
- Let the scene flow naturally from top to bottom
""",
        Orientation.LANDSCAPE: """HORIZONTAL COMPOSITION:
- Wide, panoramic layout
- Spread elements evenly from left to right
- Avoid crowded or compressed areas
""",
        Orientation.AUTO: """AUTO COMPOSITION:
- Naturally fill the entire page
- Extend the scene with simple environment elements
- No empty corners, no clutter
"""
    }

    IDEA_TO_COLORING_TEMPLATE = """Create a children's coloring page based on this idea:

{idea}

{age_mode_prompt}

COMPOSITION:
{orientation_prompt}

The illustration should feel joyful, imaginative,
and visually pleasing for children.

Ensure the result is print-ready and easy to color.
"""

    IMAGE_TO_COLORING_TEMPLATE = """Convert the provided image into a children's coloring book page.

IMPORTANT:
Do NOT trace or outline the image directly.
Redraw the scene as simplified, child-friendly line art.

GOALS:
- Preserve the main subject and overall idea of the image
- Reinterpret the scene with imagination and warmth
- Make the illustration enjoyable and easy for children to color

{age_mode_prompt}

TRANSFORMATION RULES:
- Simplify all shapes into rounded, friendly forms
- Remove all textures, shadows, lighting effects, and gradients
- Eliminate unnecessary or confusing details
- Ensure all outlines are clean, smooth, and fully closed
- Keep line thickness uniform throughout the illustration
- Avoid overlapping lines, tight spaces, or tiny decorative details
- All coloring areas must be large and comfortable to fill

COMPOSITION:
{orientation_prompt}

FINAL REQUIREMENTS:
- Black outlines on a pure white background only
- No filled black areas, no shading, no grayscale
- No text, letters, numbers, logos, or symbols
- The result must be print-ready and safe for children to color
"""

    def build_text_prompts(self, idea_input: IdeaInput) -> Dict[str, Any]:
        user_prompt = self.IDEA_TO_COLORING_TEMPLATE.format(
            idea=idea_input.idea.strip(),
            age_mode_prompt=self.AGE_MODE_PROMPTS[idea_input.age_mode],
            orientation_prompt=self.ORIENTATION_PROMPTS[idea_input.orientation],
        )
        return {
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "meta": {
                "mode": "text",
                "age_mode": idea_input.age_mode.value,
                "orientation": idea_input.orientation.value,
                "idea": idea_input.idea,
            }
        }

    def build_image_prompts(self, age_mode: AgeMode, orientation: Orientation) -> Dict[str, Any]:
        user_prompt = self.IMAGE_TO_COLORING_TEMPLATE.format(
            age_mode_prompt=self.AGE_MODE_PROMPTS[age_mode],
            orientation_prompt=self.ORIENTATION_PROMPTS[orientation],
        )
        return {
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "meta": {
                "mode": "image",
                "age_mode": age_mode.value,
                "orientation": orientation.value,
            }
        }

# ============================================================
# OpenRouter Generator（system/user 分离）
# ============================================================

class OpenRouterImageGenerator:
    """
    OpenRouter 图像生成器
    - 所有请求走 /chat/completions
    - messages: system + user 分离
    """

    SUPPORTED_MODELS = {
        "gpt5-image": "openai/gpt-5-image",
        "gpt5-image-mini": "openai/gpt-5-image-mini",
        "gemini-image": "google/gemini-2.5-flash-image",
        "nano-banana-pro": "google/gemini-3-pro-image-preview",
        "default": "openai/gpt-5-image-mini",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        gemini_image_size: str = "1K",
        openai_size: Optional[str] = None,
    ):
        """Create a generator.

        Args:
            gemini_image_size: Gemini image_config.image_size, one of {"1K","2K","4K"}.
            openai_size: Optional override for OpenAI GPT Image size. One of
                         {"1024x1024","1024x1536","1536x1024","auto"}.
                         If None, choose based on orientation.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENROUTER_API_KEY 环境变量或传入 api_key 参数")
        self.base_url = base_url
        self.prompt_system = ColoringPromptSystem()

        self.gemini_image_size = (gemini_image_size or "2K").upper()
        if self.gemini_image_size not in {"1K", "2K", "4K"}:
            raise ValueError("gemini_image_size 只能是 1K/2K/4K")

        self.openai_size = openai_size
        if self.openai_size is not None:
            allowed = {"1024x1024", "1024x1536", "1536x1024", "auto"}
            if self.openai_size not in allowed:
                raise ValueError("openai_size 只能是 1024x1024/1024x1536/1536x1024/auto")

    def generate_from_idea(
        self,
        idea_input: IdeaInput,
        model: str,
        save_path: str,
        timeout_sec: int = 180,
        retry: int = 1,
    ) -> Dict[str, Any]:
        prompts = self.prompt_system.build_text_prompts(idea_input)
        return self._generate(
            prompts=prompts,
            model=model,
            save_path=save_path,
            timeout_sec=timeout_sec,
            retry=retry,
            image_path=None,
        )

    def generate_from_image(
        self,
        image_path: str,
        age_mode: AgeMode,
        orientation: Orientation,
        model: str,
        save_path: str,
        timeout_sec: int = 180,
        retry: int = 1,
    ) -> Dict[str, Any]:
        prompts = self.prompt_system.build_image_prompts(age_mode, orientation)
        return self._generate(
            prompts=prompts,
            model=model,
            save_path=save_path,
            timeout_sec=timeout_sec,
            retry=retry,
            image_path=image_path,
        )

    def _generate(
        self,
        prompts: Dict[str, Any],
        model: str,
        save_path: str,
        timeout_sec: int,
        retry: int,
        image_path: Optional[str],
    ) -> Dict[str, Any]:
        model_id = self.SUPPORTED_MODELS.get(model, model)

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://coloring-generator.local",
            "X-Title": "Children Coloring Page Generator v2",
        }

        payload = self._build_payload(prompts, model_id, image_path=image_path)

        t0 = time.time()
        last_err = None
        for attempt in range(retry + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                resp.raise_for_status()
                result = resp.json()

                image_data = self._extract_image(result)
                if not image_data:
                    raise RuntimeError("模型未返回图像数据（message.images / data URL 为空）")

                self._save_image(image_data, save_path)
                return {
                    "success": True,
                    "model": model_id,
                    "elapsed_sec": round(time.time() - t0, 3),
                    "save_path": save_path,
                    "meta": prompts.get("meta", {}),
                    "user_prompt_preview": prompts["user_prompt"][:500],
                    "raw_response": result,
                }

            except Exception as e:
                last_err = str(e)
                if attempt < retry:
                    time.sleep(1.2 * (attempt + 1))
                continue

        return {
            "success": False,
            "model": model_id,
            "elapsed_sec": round(time.time() - t0, 3),
            "error": last_err,
            "meta": prompts.get("meta", {}),
            "user_prompt_preview": prompts["user_prompt"][:500],
        }

    def _build_payload(self, prompts: Dict[str, Any], model_id: str, image_path: Optional[str]) -> Dict[str, Any]:
        # ✅ system/user 分离
        system_msg = {"role": "system", "content": prompts["system_prompt"]}

        if image_path:
            # 图生图：user content 为多段（image + text）
            image_b64, mime_type = self._load_image_as_data_url(image_path)
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": prompts["user_prompt"]},
                ],
            }
        else:
            user_msg = {"role": "user", "content": prompts["user_prompt"]}

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [system_msg, user_msg],
            # OpenRouter: 必须声明输出模态，否则经常回落到默认 1:1
            "modalities": ["image", "text"],
        }

        # 尺寸/比例策略：按模型原生参数设置
        # - OpenAI GPT Image: 只能通过 size 选择固定画布（1024x1024 / 1024x1536 / 1536x1024 / auto）
        # - Gemini Image: 使用 image_config.aspect_ratio + image_config.image_size (1K/2K/4K)
        orientation = (prompts.get("meta", {}) or {}).get("orientation", "auto")

        if "gemini" in model_id.lower():
            aspect_map = {
                "portrait": "2:3",
                "landscape": "3:2",
                "auto": "2:3",  # 更贴近涂色卡竖版工作流
            }
            aspect_ratio = aspect_map.get(orientation, "2:3")

            payload["image_config"] = {
                "aspect_ratio": aspect_ratio,
                "image_size": self.gemini_image_size,  # 1K/2K/4K
            }
        else:
            # OpenAI GPT Image
            # NOTE: OpenRouter 文档对 Gemini 提供 image_config（aspect_ratio/image_size）控制比例。
            # 对 OpenAI 图像模型，经由 /chat/completions 的 provider 可能忽略 `size` 并回落到 1024x1024。
            # 如需稳定尺寸控制，请使用 Gemini 图像模型或 --api openai（官方 Images API）。
            if (self.openai_size is not None) or (orientation in ("portrait", "landscape", "auto")):
                print("[WARN] OpenRouter OpenAI image models may ignore `size` and return 1024x1024. Use a Gemini image model or --api openai for guaranteed aspect ratio.")
            if self.openai_size is not None:
                payload["size"] = self.openai_size
            else:
                if orientation == "landscape":
                    payload["size"] = "1536x1024"
                elif orientation == "portrait":
                    payload["size"] = "1024x1536"
                else:
                    payload["size"] = "1024x1536"  # auto 默认竖版

            payload["quality"] = "high"

        return payload

    def _load_image_as_data_url(self, image_path: str) -> (str, str):
        ext = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/png")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{b64}", mime_type

    def _extract_image(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract image data/url from OpenRouter chat/completions response."""
        try:
            choices = result.get("choices", [])
            if not choices:
                return None

            msg = (choices[0].get("message") or {})

            # 1) Common OpenRouter image response: message.images
            images = msg.get("images", [])
            if isinstance(images, list) and images:
                first = images[0]
                if isinstance(first, dict):
                    url = (first.get("image_url") or {}).get("url")
                    if url:
                        return url
                if isinstance(first, str):
                    return first

            content = msg.get("content")

            # 2) Some providers return multi-part content list
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url")
                        if url:
                            return url
                    if part.get("type") == "image":
                        url = (part.get("image") or {}).get("url")
                        if url:
                            return url

            # 3) content as string: data URL or http URL
            if isinstance(content, str):
                if content.startswith("http") or content.startswith("data:image"):
                    return content

            return None
        except Exception:
            return None

    def _save_image(self, image_data: str, save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if image_data.startswith("http"):
            r = requests.get(image_data, timeout=60)
            r.raise_for_status()
            raw = r.content
        elif image_data.startswith("data:image"):
            b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
            raw = base64.b64decode(b64)
        else:
            raw = base64.b64decode(image_data)

        with open(save_path, "wb") as f:
            f.write(raw)
 
class OpenAIOfficialImageGenerator:
    """OpenAI 官方 Images API 生成器（稳定支持 size 控制）

    - 文生图：POST /v1/images/generations
    - 图生图：POST /v1/images/edits

    需要环境变量：OPENAI_API_KEY
    """

    SUPPORTED_MODELS = {
        "gpt-image-1": "gpt-image-1",
        "gpt-image-1-mini": "gpt-image-1-mini",
        "gpt-image-1.5": "gpt-image-1.5",
        "dall-e-3": "dall-e-3",
        "dall-e-2": "dall-e-2",
        "default": "gpt-image-1",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_size: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数（使用 --api openai 时必需）")

        self.base_url = base_url.rstrip("/")
        self.prompt_system = ColoringPromptSystem()

        self.openai_size = openai_size
        if self.openai_size is not None:
            allowed = {"1024x1024", "1024x1536", "1536x1024"}
            if self.openai_size not in allowed:
                raise ValueError("openai_size 只能是 1024x1024/1024x1536/1536x1024")

    def generate_from_idea(
        self,
        idea_input: IdeaInput,
        model: str,
        save_path: str,
        timeout_sec: int = 180,
        retry: int = 1,
    ) -> Dict[str, Any]:
        prompts = self.prompt_system.build_text_prompts(idea_input)
        return self._generate(
            prompts=prompts,
            model=model,
            save_path=save_path,
            timeout_sec=timeout_sec,
            retry=retry,
            image_path=None,
        )

    def generate_from_image(
        self,
        image_path: str,
        age_mode: AgeMode,
        orientation: Orientation,
        model: str,
        save_path: str,
        timeout_sec: int = 180,
        retry: int = 1,
    ) -> Dict[str, Any]:
        prompts = self.prompt_system.build_image_prompts(age_mode, orientation)
        return self._generate(
            prompts=prompts,
            model=model,
            save_path=save_path,
            timeout_sec=timeout_sec,
            retry=retry,
            image_path=image_path,
        )

    def _choose_size(self, orientation: str) -> str:
        if self.openai_size is not None:
            return self.openai_size
        if orientation == "landscape":
            return "1536x1024"
        # portrait / auto 默认竖版
        return "1024x1536"

    def _generate(
        self,
        prompts: Dict[str, Any],
        model: str,
        save_path: str,
        timeout_sec: int,
        retry: int,
        image_path: Optional[str],
    ) -> Dict[str, Any]:
        model_id = self.SUPPORTED_MODELS.get(model, model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        meta = (prompts.get("meta", {}) or {})
        orientation = meta.get("orientation", "auto")
        size = self._choose_size(orientation)

        request_info = {
            "requested_size": size,
            "endpoint": None,
        }

        t0 = time.time()
        last_err = None

        for attempt in range(retry + 1):
            try:
                if image_path is None:
                    # 文生图
                    endpoint = f"{self.base_url}/images/generations"
                    request_info["endpoint"] = endpoint
                    payload = {
                        "model": model_id,
                        "prompt": prompts["user_prompt"],
                        "n": 1,
                        "size": size,
                        "quality": "high",
                    }
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                else:
                    # 图生图（edits，multipart/form-data）
                    endpoint = f"{self.base_url}/images/edits"
                    request_info["endpoint"] = endpoint
                    data = {
                        "model": model_id,
                        "prompt": prompts["user_prompt"],
                        "n": "1",
                        "size": size,
                        "quality": "high",
                    }
                    with open(image_path, "rb") as f:
                        files = {
                            "image": (os.path.basename(image_path), f, "application/octet-stream"),
                        }
                        resp = requests.post(endpoint, headers=headers, data=data, files=files, timeout=timeout_sec)

                resp.raise_for_status()
                result = resp.json()
                # Debug hint if gateway returns a non-standard schema
                if not isinstance(result, dict):
                    raise RuntimeError(f"OpenAI Images API 返回了非 JSON 对象: {type(result)}")

                image_data = self._extract_image(result)
                if not image_data:
                    raise RuntimeError(
                        "OpenAI Images API 未返回 data[0].b64_json 或 url; "
                        f"response keys={list(result.keys())}; "
                        f"snippet={json.dumps(result, ensure_ascii=False)[:400]}"
                    )

                self._save_image(image_data, save_path)

                # 可选：打印保存后的图片尺寸，快速确认是否仍为 1024x1024
                try:
                    from PIL import Image
                    with Image.open(save_path) as im:
                        print("  SAVED PNG SIZE:", im.size)
                except Exception:
                    pass

                return {
                    "success": True,
                    "model": model_id,
                    "elapsed_sec": round(time.time() - t0, 3),
                    "save_path": save_path,
                    "meta": meta,
                    "request_info": request_info,
                    "user_prompt_preview": prompts["user_prompt"][:500],
                    "raw_response": result,
                }

            except Exception as e:
                last_err = str(e)
                if attempt < retry:
                    time.sleep(1.2 * (attempt + 1))
                continue

        return {
            "success": False,
            "model": model_id,
            "elapsed_sec": round(time.time() - t0, 3),
            "error": last_err,
            "meta": meta,
            "request_info": request_info,
            "user_prompt_preview": prompts["user_prompt"][:500],
        }

    def _extract_image(self, result: Dict[str, Any]) -> Optional[str]:
        try:
            data = result.get("data")
            if isinstance(data, list) and data:
                first = data[0] or {}
                b64_json = first.get("b64_json")
                if b64_json:
                    return b64_json
                url = first.get("url")
                if url:
                    return url
            return None
        except Exception:
            return None

    def _save_image(self, image_data: str, save_path: str) -> None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if image_data.startswith("http"):
            r = requests.get(image_data, timeout=60)
            r.raise_for_status()
            raw = r.content
        elif image_data.startswith("data:image"):
            b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
            raw = base64.b64decode(b64)
        else:
            # OpenAI Images API 常见：直接给 b64_json
            raw = base64.b64decode(image_data)

        with open(save_path, "wb") as f:
            f.write(raw)


# ============================================================
# 批量测试集 & Runner
# ============================================================

def default_test_set() -> List[IdeaInput]:
    return [
        IdeaInput("A friendly dinosaur having a picnic in a sunny forest with butterflies", AgeMode.KIDS, Orientation.AUTO),
        IdeaInput("A smiling teddy bear riding a tiny train through a flower garden", AgeMode.TODDLER, Orientation.AUTO),
        IdeaInput("A magical underwater world with playful fish, a friendly octopus, and sea plants", AgeMode.KIDS, Orientation.PORTRAIT),
        IdeaInput("A cute puppy flying a kite in a park with fluffy clouds", AgeMode.TODDLER, Orientation.PORTRAIT),
        IdeaInput("A friendly dragon carefully watering a secret garden with big flowers", AgeMode.KIDS, Orientation.AUTO),
        IdeaInput("A happy robot helping animals build a tree house", AgeMode.KIDS, Orientation.AUTO),
        IdeaInput("A little cat exploring a pumpkin village with cozy houses", AgeMode.KIDS, Orientation.PORTRAIT),
        IdeaInput("A smiling sun and a rainbow over a field with simple flowers", AgeMode.TODDLER, Orientation.PORTRAIT),
    ]


def run_batch(gen: OpenRouterImageGenerator, cases: List[IdeaInput], model: str, out_dir: str, retry: int) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for idx, idea_input in enumerate(cases, start=1):
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{idx:03d}_{idea_input.age_mode.value}_{idea_input.orientation.value}_{ts}"
        png_path = str(Path(out_dir) / f"{stem}.png")
        json_path = str(Path(out_dir) / f"{stem}.json")

        print(f"\n[{idx}/{len(cases)}] model={model} age={idea_input.age_mode.value} orient={idea_input.orientation.value}")
        print(f"  idea: {idea_input.idea}")

        res = gen.generate_from_idea(
            idea_input=idea_input,
            model=model,
            save_path=png_path,
            retry=retry,
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        if res["success"]:
            print(f"  OK   -> {res['save_path']} ({res['elapsed_sec']}s)")
        else:
            print(f"  FAIL -> {res['error']} ({res['elapsed_sec']}s)")
        results.append(res)

    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if not r.get("success")),
        "model": model,
        "out_dir": out_dir,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(str(Path(out_dir) / "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ============================================================
# 交互模式（仅 idea）
# ============================================================

def interactive_mode(gen: OpenRouterImageGenerator, model: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    age_mode = AgeMode.KIDS
    orientation = Orientation.PORTRAIT

    print("\n儿童涂色卡生成器 v2（单输入 idea）")
    print("命令：")
    print("  toddler / kids            切换年龄")
    print("  portrait / landscape / auto  切换方向")
    print("  quit                      退出\n")

    while True:
        try:
            prompt = input(f"[{age_mode.value}/{orientation.value}] Idea> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("Bye.")
                break
            if prompt.lower() == "toddler":
                age_mode = AgeMode.TODDLER
                print("已切换 toddler")
                continue
            if prompt.lower() == "kids":
                age_mode = AgeMode.KIDS
                print("已切换 kids")
                continue
            if prompt.lower() in ("portrait", "landscape", "auto"):
                orientation = Orientation(prompt.lower())
                print(f"已切换 {orientation.value}")
                continue

            idea_input = IdeaInput(prompt, age_mode, orientation)
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = str(Path(out_dir) / f"interactive_{age_mode.value}_{orientation.value}_{ts}.png")

            res = gen.generate_from_idea(idea_input, model=model, save_path=save_path, retry=1)
            if res["success"]:
                print(f"生成成功: {save_path}\n")
            else:
                print(f"生成失败: {res['error']}\n")

        except KeyboardInterrupt:
            print("\nBye.")
            break


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nano-banana-pro", help="模型别名或完整 model id")
    p.add_argument("--out", default="output_v2", help="输出目录")
    p.add_argument("--retry", type=int, default=1, help="每条用例重试次数")
    p.add_argument("--api", default="openrouter", choices=["openrouter", "openai"],
                   help="调用后端：openrouter=OpenRouter chat/completions；openai=OpenAI 官方 Images API（需要 OPENAI_API_KEY）")

    # 文生图（idea）
    p.add_argument("--generate", type=str, default=None, help="输入一句话 idea 生成涂色卡")
    # 图生图
    p.add_argument("--image2color", type=str, default=None, help="输入图片路径 -> 输出涂色卡")

    # batch / interactive
    p.add_argument("--batch", action="store_true", help="跑默认批量测试集")
    p.add_argument("--interactive", action="store_true", help="交互模式")

    # 控制变量
    p.add_argument("--toddler", action="store_true", help="toddler 模式")
    p.add_argument("--kids", action="store_true", help="kids 模式（默认）")
    p.add_argument("--portrait", action="store_true", help="竖版")
    p.add_argument("--landscape", action="store_true", help="横版")
    p.add_argument("--auto", action="store_true", help="自动构图")

    # 输出尺寸控制（不同模型支持不同方式）
    p.add_argument("--gemini-image-size", default="1K", choices=["1K", "2K", "4K"],
                   help="Gemini 系列 image_config.image_size（默认 1K）")
    p.add_argument("--openai-size", default=None, choices=["1024x1024", "1024x1536", "1536x1024", "auto"],
                   help="OpenAI Images API 的 size 覆盖值；在 OpenRouter+OpenAI 模型上可能被忽略。不传则根据 orientation 自动选择")
    p.add_argument("--openai-base-url", default=None,
                   help="OpenAI Images API base_url 覆盖值，例如 https://api.openai.com/v1 或 https://api.bianxie.ai/v1。也可用环境变量 OPENAI_BASE_URL")

    return p.parse_args()


def main():
    args = parse_args()

    age_mode = AgeMode.TODDLER if args.toddler else AgeMode.KIDS
    if args.kids:
        age_mode = AgeMode.KIDS

    orientation = Orientation.PORTRAIT
    if args.landscape:
        orientation = Orientation.LANDSCAPE
    elif args.auto:
        orientation = Orientation.AUTO
    elif args.portrait:
        orientation = Orientation.PORTRAIT

    if args.api == "openai":
        base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        gen = OpenAIOfficialImageGenerator(base_url=base_url, openai_size=args.openai_size)
    else:
        gen = OpenRouterImageGenerator(gemini_image_size=args.gemini_image_size, openai_size=args.openai_size)

    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.batch:
        cases = default_test_set()
        run_batch(gen, cases, args.model, out_dir, args.retry)
        return

    if args.interactive:
        interactive_mode(gen, args.model, out_dir)
        return

    if args.generate:
        idea_input = IdeaInput(args.generate, age_mode, orientation)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = str(Path(out_dir) / f"idea_{age_mode.value}_{orientation.value}_{ts}.png")
        res = gen.generate_from_idea(idea_input, args.model, save_path, retry=args.retry)
        print(json.dumps({
            "success": res["success"],
            "save_path": res.get("save_path"),
            "error": res.get("error"),
            "model": res.get("model"),
        }, ensure_ascii=False, indent=2))
        return

    if args.image2color:
        image_path = args.image2color
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = str(Path(out_dir) / f"img2color_{age_mode.value}_{orientation.value}_{ts}.png")
        res = gen.generate_from_image(
            image_path=image_path,
            age_mode=age_mode,
            orientation=orientation,
            model=args.model,
            save_path=save_path,
            retry=args.retry,
        )
        print(json.dumps({
            "success": res["success"],
            "save_path": res.get("save_path"),
            "error": res.get("error"),
            "model": res.get("model"),
        }, ensure_ascii=False, indent=2))
        return

    # 默认：提示用法
    print("用法示例：")
    print("  python coloring_generator_v2.py --batch --model gpt5-image-mini --out output_batch")
    print("  python coloring_generator_v2.py --generate \"A friendly dragon flying over a castle\" --kids --portrait --model nano-banana-pro")
    print("  python coloring_generator_v2.py --image2color ./input.jpg --kids --portrait --model gpt5-image-mini")
    print("  python coloring_generator.py --api openai --generate \"A friendly dragon\" --portrait --model gpt-image-1.5 --openai-size 1024x1536")
    print("  python coloring_generator_v2.py --interactive")


if __name__ == "__main__":
    main()