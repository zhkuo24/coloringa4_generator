#!/usr/bin/env python3
"""
儿童涂色卡生成器 v3（统一 API 架构）
- 支持两种 API 格式：/chat/completions 和 /images/generations
- 根据模型配置自动选择 API 类型
- 支持切换不同 API 平台（通过 base_url）

环境变量：
  API_KEY=xxxx
  API_BASE_URL=https://yunwu.ai/v1

用法示例：
  python coloring_generator.py --generate "A friendly dragon" --model gpt-image-1-mini
  python coloring_generator.py --generate "机器人在球场打球" --auto
  python coloring_generator.py --interactive
"""

import os
import re
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


class APIType(Enum):
    """API 接口格式（都是文生图，但不同模型使用不同接口）"""
    CHAT_COMPLETIONS = "chat"      # /v1/chat/completions（Gemini 等模型使用）
    IMAGES_GENERATIONS = "images"  # /v1/images/generations（GPT-Image、DALL-E 等模型使用）


@dataclass
class IdeaInput:
    """单输入 idea 模式"""
    idea: str
    age_mode: AgeMode = AgeMode.KIDS
    orientation: Orientation = Orientation.AUTO


# ============================================================
# 模型配置
# ============================================================

# 默认分辨率配置（GPT-Image 等模型使用）
DEFAULT_SIZES = {"portrait": "1024x1536", "landscape": "1536x1024"}

# Gemini 模型专用 aspectRatio 配置
GEMINI_ASPECT_RATIOS = {"portrait": "2:3", "landscape": "4:3"}

MODEL_CONFIG = {
    "gpt-image-1": {"model_id": "gpt-image-1", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "OPENAI_API_KEY"},
    "gpt-image-1-mini": {"model_id": "gpt-image-1-mini", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "OPENAI_API_KEY"},
    "gpt-image-1.5": {"model_id": "gpt-image-1.5", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "OPENAI_API_KEY"},
    "gemini-image": {"model_id": "gemini-3-pro-image-preview", "api_type": APIType.CHAT_COMPLETIONS, "api_key_env": "GEMINI_API_KEY"},
    "z-image": {"model_id": "z-image", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "API_KEY"},
    "default": {"model_id": "gpt-image-1", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "API_KEY"},
}

# ============================================================
# Prompt System（融合版，生产级）
# ============================================================

class ColoringPromptSystem:
    """
    Prompt 架构：
      - system_prompt 固定：质量/印刷/安全底线
      - user_prompt 动态：idea + age + orientation
    """

    SYSTEM_PROMPT = """You are a professional children's coloring-book line-art illustrator.
You create print-ready black-and-white coloring pages that are fun, whimsical, and EASY for children to color.

NON-NEGOTIABLE OUTPUT FORMAT
- Output MUST be a single coloring-page illustration: pure black outlines on pure white background.
- NO text of any kind: no letters, numbers, logos, watermarks, symbols, labels, signs.
- NO grayscale: no shading, hatching, gradients, textures, or filled black areas.
- Uniform line weight across the entire page (print-friendly).
- Lines must be clean, smooth, and continuous; ALL regions must be fully closed.
- Avoid overlaps, doubled lines, tangled lines, and intersections that create tiny slivers.
- Avoid tiny details: no thin corridors, small holes, cramped patterns, or micro-decorations.

CHILD COLORING USABILITY (HIGHEST PRIORITY)
- Every enclosed region must be comfortably colorable.
- Use large, simple shapes with generous spacing.
- Prefer rounded, friendly silhouettes.
- Keep the scene readable at a glance (clear subject + clear background).

IMAGINATION & CHILD APPEAL
- Mood: cheerful, playful, storybook-like.
- Characters: friendly faces, simple expressions, gentle humor.
- Imagination should come from the scene idea + playful props + composition,
  NOT from dense detail or texture.

COMPOSITION & PRINT LAYOUT
- Balanced composition with clear white margins.
- Fill the page naturally: no empty corners, no clutter.
- Foreground / midground / background separation using simple shapes and spacing.
- Include a simple “ground” and “sky” cue when appropriate.

INTERNAL WORKFLOW (DO NOT OUTPUT THESE STEPS)
1) Identify main subject(s) and 2–3 supporting elements that reinforce the story.
2) Choose a clear focal point and keep background minimal but relevant.
3) Ensure all lines close; remove any elements that would create cramped regions.
4) Final self-check against all rules; if any rule is violated, simplify and redraw internally.

FINAL SELF-CHECK BEFORE YOU FINISH
- Closed shapes everywhere, no gaps.
- Consistent line thickness.
- No filled blacks, shading, gradients, textures, or text.
- Age-appropriate complexity and spacious coloring areas.
"""

    AGE_MODE_PROMPTS = {
        AgeMode.TODDLER: """[AGE MODE: TODDLER / PRESCHOOL]
Target complexity: VERY LOW
- 2–5 large, friendly objects total (including background cues)
- Very large, open coloring regions
- Very wide spacing; avoid small interior holes and tiny decorations
- No patterns; no dense backgrounds; no micro-details
- Instantly recognizable silhouettes and faces

Imagination style:
Warm, comforting, cute, and simple.
One clear "story moment" with gentle whimsy.
""",
        AgeMode.KIDS: """[AGE MODE: KIDS]
Target complexity: MODERATE
- 5–10 clear objects total (foreground + a few background elements)
- Medium-to-large coloring regions (avoid micro details)
- Simple, BIG decorative elements allowed only if large and sparse
- Background must directly support the theme, not compete with the subject
- Characters may show action and emotion; keep poses readable

Imagination style:
Playful and adventurous.
Add 1–2 whimsical props that logically belong to the idea (no random extras).
"""
    }

    IDEA_TO_COLORING_TEMPLATE = """Create ONE print-ready children's coloring page illustration from this idea:

IDEA:
{idea}

{age_mode_prompt}

STORYBOOK CHARM (keep it simple):
- Depict a single clear “story moment” that feels joyful.
- Add 1–2 whimsical details that logically belong to the idea (no random extras).
- Keep the scene uncluttered with spacious coloring areas.

STRICT COLORING-PAGE RULES (must follow):
- Pure black outlines on pure white background only.
- Closed shapes everywhere; smooth continuous lines; uniform line weight.
- No shading, no hatching, no gradients, no textures, no filled black areas.
- No text/letters/numbers/logos/symbols.
- No overlaps/tangled/doubled lines; no tiny gaps or cramped regions.

Deliver a clean, professional coloring-book style page that children will love.
"""

    def build_text_prompts(self, idea_input: IdeaInput) -> Dict[str, Any]:
        user_prompt = self.IDEA_TO_COLORING_TEMPLATE.format(
            idea=idea_input.idea.strip(),
            age_mode_prompt=self.AGE_MODE_PROMPTS[idea_input.age_mode]
        )
        return {
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "meta": {
                "mode": "text",
                "age_mode": idea_input.age_mode.value,
                "idea": idea_input.idea,
            }
        }


# ============================================================
# Orientation Detector（使用便宜的文本模型判断横竖版）
# ============================================================

class OrientationDetector:
    """使用 LLM 自动判断涂色卡最佳布局方向"""

    DEFAULT_MODEL = "gpt-4o-mini"

    DETECTION_PROMPT = """Determine the best page orientation (PORTRAIT or LANDSCAPE) for a children's coloring book illustration based on the likely composition.

Idea: {idea}

Decide by composition, not by object name.

PORTRAIT if the main subject is a single tall/vertical focus or needs top-to-bottom framing:
- one main character centered (full body), standing pose
- tall objects: tower, rocket, lighthouse, skyscraper, tree, giraffe, waterfall
- scenes emphasizing height, climbing, stacked elements, vertical movement

LANDSCAPE if the scene needs left-to-right space or wide background:
- wide environments with horizon/skyline: beach/ocean, desert, mountains, sunset, city skyline
- long paths/fields: road/highway, racetrack, runway, sports field/court, river across the scene
- multiple subjects arranged side-by-side across the page, group activities spread horizontally
- panoramic or “wide view” composition

Conflict rule:
- If there is a single dominant tall subject that should be the focal point, choose PORTRAIT.
- Otherwise, if the story requires showing wide context or multiple elements across, choose LANDSCAPE.
- If still uncertain, choose PORTRAIT.

Respond with exactly one word: PORTRAIT or LANDSCAPE."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 API_KEY 环境变量或传入 api_key 参数")
        self.base_url = (base_url or os.getenv("API_BASE_URL", "https://yunwu.ai/v1")).rstrip("/")
        self.model = model or self.DEFAULT_MODEL

    def detect(self, idea: str) -> Orientation:
        """基于 idea 内容自动判断 portrait/landscape"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": self.DETECTION_PROMPT.format(idea=idea)}
            ],
            "max_tokens": 20,
            "temperature": 0,
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            answer = result["choices"][0]["message"]["content"].strip().upper()

            if "LANDSCAPE" in answer:
                print(f"[OrientationDetector] idea=\"{idea}\" → LANDSCAPE")
                return Orientation.LANDSCAPE
            else:
                print(f"[OrientationDetector] idea=\"{idea}\" → PORTRAIT")
                return Orientation.PORTRAIT

        except Exception as e:
            print(f"[OrientationDetector] 检测失败，默认使用 PORTRAIT: {e}")
            return Orientation.PORTRAIT


# ============================================================
# 统一的图像生成器
# ============================================================

class ImageGenerator:
    """
    统一的图像生成器，支持多种 API 格式和平台

    特点：
    1. 根据模型配置自动选择 API 类型（chat/completions 或 images/generations）
    2. 支持切换不同 API 平台（通过 base_url）
    3. 提示词系统保持不变
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("API_BASE_URL", "https://yunwu.ai/v1")).rstrip("/")
        self.prompt_system = ColoringPromptSystem()

    def get_model_config(self, model: str) -> Dict[str, Any]:
        """获取模型配置"""
        if model in MODEL_CONFIG:
            return MODEL_CONFIG[model]
        return {"model_id": model, "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "API_KEY"}

    def _get_api_key(self, config: Dict[str, Any]) -> str:
        """根据模型配置获取对应的 API KEY"""
        env_name = config.get("api_key_env", "API_KEY")
        key = os.getenv(env_name)
        if not key:
            raise ValueError(f"请设置 {env_name} 环境变量")
        return key

    def generate_from_idea(
        self,
        idea_input: IdeaInput,
        model: str,
        save_path: str,
        timeout_sec: int = 180,
        retry: int = 1,
    ) -> Dict[str, Any]:
        """文生图主入口"""
        prompts = self.prompt_system.build_text_prompts(idea_input)
        config = self.get_model_config(model)
        api_key = self._get_api_key(config)
        # 从 idea_input 获取 orientation（由 OrientationDetector 判断）
        orientation = idea_input.orientation.value  # "portrait" 或 "landscape"

        if config["api_type"] == APIType.CHAT_COMPLETIONS:
            return self._generate_via_chat(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)
        else:
            return self._generate_via_images(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)

    # ============================================================
    # Chat Completions API (/chat/completions)
    # ============================================================

    def _generate_via_chat(
        self, prompts: Dict[str, Any], model_id: str, api_key: str, orientation: str, save_path: str, timeout_sec: int, retry: int
    ) -> Dict[str, Any]:
        """使用 /chat/completions 接口生成图像"""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = self._build_chat_payload(prompts, model_id, orientation)

        t0 = time.time()
        last_err = None

        for attempt in range(retry + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                resp.raise_for_status()
                result = resp.json()

                image_data = self._extract_image_from_chat(result)
                if not image_data:
                    raise RuntimeError("模型未返回图像数据")

                self._save_image(image_data, save_path)
                return {
                    "success": True,
                    "model": model_id,
                    "api_type": "chat",
                    "elapsed_sec": round(time.time() - t0, 3),
                    "save_path": save_path,
                    "meta": prompts.get("meta", {}),
                    "user_prompt_preview": prompts["user_prompt"][:500],
                }

            except Exception as e:
                last_err = str(e)
                if attempt < retry:
                    time.sleep(1.2 * (attempt + 1))
                continue

        return {
            "success": False,
            "model": model_id,
            "api_type": "chat",
            "elapsed_sec": round(time.time() - t0, 3),
            "error": last_err,
            "meta": prompts.get("meta", {}),
            "user_prompt_preview": prompts["user_prompt"][:500],
        }

    def _build_chat_payload(self, prompts: Dict[str, Any], model_id: str, orientation: str) -> Dict[str, Any]:
        """构建 chat/completions 请求 payload（Gemini 模型使用 imageConfig）"""
        aspect_ratio = GEMINI_ASPECT_RATIOS.get(orientation, "2:3")
        return {
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": prompts["user_prompt"]},
            ],
            "modalities": ["image", "text"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": "2K",
            },
        }

    def _extract_image_from_chat(self, result: Dict[str, Any]) -> Optional[str]:
        """从 chat/completions 响应中提取图像"""
        try:
            choices = result.get("choices", [])
            if not choices:
                return None

            msg = (choices[0].get("message") or {})

            # 1) message.images 列表
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

            # 2) 多部分内容列表
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

            # 3) content 为字符串（data URL 或 http URL）
            if isinstance(content, str):
                if content.startswith("http") or content.startswith("data:image"):
                    return content

                # 4) Markdown 格式图像: ![image](data:image/xxx;base64,...)
                match = re.search(r'!\[.*?\]\((data:image/[^)]+)\)', content)
                if match:
                    return match.group(1)
                # 也匹配 http URL
                match = re.search(r'!\[.*?\]\((https?://[^)]+)\)', content)
                if match:
                    return match.group(1)

            return None
        except Exception:
            return None

    # ============================================================
    # Images API (/images/generations)
    # ============================================================

    def _generate_via_images(
        self, prompts: Dict[str, Any], model_id: str, api_key: str, orientation: str, save_path: str, timeout_sec: int, retry: int
    ) -> Dict[str, Any]:
        """使用 /images/generations 接口生成图像"""
        endpoint = f"{self.base_url}/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload = {
            "model": model_id,
            "prompt": f"{prompts['system_prompt']}\n\n{prompts['user_prompt']}",
            "n": 1,
            "size": DEFAULT_SIZES.get(orientation, DEFAULT_SIZES["portrait"]),
            "quality": "high",
        }

        t0 = time.time()
        last_err = None

        for attempt in range(retry + 1):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                resp.raise_for_status()
                result = resp.json()

                image_data = self._extract_image_from_images(result)
                if not image_data:
                    raise RuntimeError(f"API 未返回图像数据: {json.dumps(result, ensure_ascii=False)[:400]}")

                self._save_image(image_data, save_path)

                # 打印保存后的图片尺寸
                self._print_saved_image_size(save_path)

                return {
                    "success": True,
                    "model": model_id,
                    "api_type": "images",
                    "elapsed_sec": round(time.time() - t0, 3),
                    "save_path": save_path,
                    "meta": prompts.get("meta", {}),
                    "user_prompt_preview": prompts["user_prompt"][:500],
                }

            except Exception as e:
                last_err = str(e)
                if attempt < retry:
                    time.sleep(1.2 * (attempt + 1))
                continue

        return {
            "success": False,
            "model": model_id,
            "api_type": "images",
            "elapsed_sec": round(time.time() - t0, 3),
            "error": last_err,
            "meta": prompts.get("meta", {}),
            "user_prompt_preview": prompts["user_prompt"][:500],
        }

    def _extract_image_from_images(self, result: Dict[str, Any]) -> Optional[str]:
        """从 images API 响应中提取图像"""
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

    # ============================================================
    # 工具方法
    # ============================================================

    def _save_image(self, image_data: str, save_path: str) -> None:
        """保存图像到文件"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if image_data.startswith("http"):
            r = requests.get(image_data, timeout=60)
            r.raise_for_status()
            raw = r.content
        elif image_data.startswith("data:image"):
            b64 = image_data.split(",", 1)[1] if "," in image_data else image_data
            raw = base64.b64decode(b64)
        else:
            # 直接给 base64
            raw = base64.b64decode(image_data)

        with open(save_path, "wb") as f:
            f.write(raw)

    def _print_saved_image_size(self, save_path: str) -> None:
        """打印保存后的图片尺寸"""
        try:
            from PIL import Image
            with Image.open(save_path) as im:
                print(f"  SAVED PNG SIZE: {im.size}")
        except Exception:
            pass


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


def run_batch(gen: ImageGenerator, cases: List[IdeaInput], model: str, out_dir: str, retry: int) -> Dict[str, Any]:
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
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="儿童涂色卡生成器 v3（统一 API 架构）")
    p.add_argument("--model", default="gpt-image-1-mini", help="模型别名或完整 model id")
    p.add_argument("--out", default="output_v3", help="输出目录")
    p.add_argument("--retry", type=int, default=1, help="每条用例重试次数")
    p.add_argument("--base-url", default=None, help="API base URL（默认从环境变量 API_BASE_URL 读取）")

    # 文生图（idea）
    p.add_argument("--generate", type=str, default=None, help="输入一句话 idea 生成涂色卡")

    # batch / interactive
    p.add_argument("--batch", action="store_true", help="跑默认批量测试集")
    p.add_argument("--interactive", action="store_true", help="交互模式")

    # 控制变量
    p.add_argument("--toddler", action="store_true", help="toddler 模式")
    p.add_argument("--kids", action="store_true", help="kids 模式（默认）")
    p.add_argument("--portrait", action="store_true", help="竖版")
    p.add_argument("--landscape", action="store_true", help="横版")
    p.add_argument("--auto", action="store_true", help="自动检测横竖版（使用 GPT-4o Mini 判断）")

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

    # 创建生成器
    gen = ImageGenerator(base_url=args.base_url)

    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.batch:
        cases = default_test_set()
        run_batch(gen, cases, args.model, out_dir, args.retry)
        return


    if args.generate:
        # 如果是 AUTO 模式，使用 OrientationDetector 自动检测
        if orientation == Orientation.AUTO:
            detector = OrientationDetector()
            orientation = detector.detect(args.generate)

        idea_input = IdeaInput(args.generate, age_mode, orientation)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_path = str(Path(out_dir) / f"idea_{age_mode.value}_{orientation.value}_{ts}.png")
        res = gen.generate_from_idea(idea_input, args.model, save_path, retry=args.retry)
        print(json.dumps({
            "success": res["success"],
            "save_path": res.get("save_path"),
            "error": res.get("error"),
            "model": res.get("model"),
            "api_type": res.get("api_type"),
        }, ensure_ascii=False, indent=2))
        return

    # 默认：提示用法
    print("用法示例：")
    print("  python coloring_generator.py --generate \"A friendly dragon\" --model gpt-image-1-mini")
    print("  python coloring_generator.py --generate \"机器人在球场打球\" --auto")
    print("  python coloring_generator.py --generate \"一棵圣诞树\" --model gemini-image")
    print("  python coloring_generator.py --batch")
    print("  python coloring_generator.py --interactive")
    print("\n支持的模型：")
    for name, config in MODEL_CONFIG.items():
        if name != "default":
            print(f"  {name}: {config['model_id']} ({config['api_type'].value})")


if __name__ == "__main__":
    main()
