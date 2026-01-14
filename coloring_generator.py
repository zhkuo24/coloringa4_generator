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
# 模型配置（包含 API 类型）
# ============================================================

MODEL_CONFIG = {
    # GPT-Image 系列 - 使用 images API
    "gpt-image-1": {
        "model_id": "gpt-image-1",
        "api_type": APIType.IMAGES_GENERATIONS,
    },
    "gpt-image-1-mini": {
        "model_id": "gpt-image-1-mini",
        "api_type": APIType.IMAGES_GENERATIONS,
    },
    "gpt-image-1.5": {
        "model_id": "gpt-image-1.5",
        "api_type": APIType.IMAGES_GENERATIONS,
    },
    # Gemini 系列 - 使用 chat API（支持 modalities）
    "gemini-image": {
        "model_id": "gemini-3-pro-image-preview",
        "api_type": APIType.CHAT_COMPLETIONS,
    },
    # z-image - 使用 images API
    "z-image": {
        "model_id": "z-image",
        "api_type": APIType.IMAGES_GENERATIONS,
    },

    # 默认模型
    "default": {
        "model_id": "gpt-image-1",
        "api_type": APIType.IMAGES_GENERATIONS,
    },
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
- Background should directly relate to the requested scene
- Characters may show action and emotion

Imagination style:
Playful and adventurous.
Focus on the specific theme requested by the user.
Add only elements that logically belong to the scene.
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
- Naturally fill the entire page with the main subject
- Focus on the requested elements, avoid adding unrelated objects
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


# ============================================================
# Orientation Detector（使用便宜的文本模型判断横竖版）
# ============================================================

class OrientationDetector:
    """使用 LLM 自动判断涂色卡最佳布局方向"""

    DEFAULT_MODEL = "gpt-4o-mini"

    DETECTION_PROMPT = """判断这个儿童涂色卡的最佳布局方向。

创意内容: {idea}

只回答一个词: PORTRAIT 或 LANDSCAPE

判断规则:
- 宽/横向场景（球场、海洋、公路、多人并排、赛车、火车）→ LANDSCAPE
- 高/纵向场景（树、塔、火箭、站立人物、长颈鹿、摩天大楼）→ PORTRAIT
- 不确定时默认 → PORTRAIT
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("API_KEY")
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

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("请设置 API_KEY 环境变量或传入 api_key 参数")
        self.base_url = (base_url or os.getenv("API_BASE_URL", "https://yunwu.ai/v1")).rstrip("/")
        self.prompt_system = ColoringPromptSystem()

    def get_model_config(self, model: str) -> Dict[str, Any]:
        """获取模型配置，支持直接传入完整 model_id"""
        if model in MODEL_CONFIG:
            return MODEL_CONFIG[model]
        # 如果不在配置中，假设是完整的 model_id，默认使用 images API
        return {
            "model_id": model,
            "api_type": APIType.IMAGES_GENERATIONS,
        }

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

        if config["api_type"] == APIType.CHAT_COMPLETIONS:
            return self._generate_via_chat(
                prompts=prompts,
                model_id=config["model_id"],
                save_path=save_path,
                timeout_sec=timeout_sec,
                retry=retry,
            )
        else:
            return self._generate_via_images(
                prompts=prompts,
                model_id=config["model_id"],
                save_path=save_path,
                timeout_sec=timeout_sec,
                retry=retry,
            )

    # ============================================================
    # Chat Completions API (/chat/completions)
    # ============================================================

    def _generate_via_chat(
        self,
        prompts: Dict[str, Any],
        model_id: str,
        save_path: str,
        timeout_sec: int,
        retry: int,
    ) -> Dict[str, Any]:
        """使用 /chat/completions 接口生成图像"""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = self._build_chat_payload(prompts, model_id)

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

    def _build_chat_payload(
        self,
        prompts: Dict[str, Any],
        model_id: str,
    ) -> Dict[str, Any]:
        """构建 chat/completions 请求 payload"""
        system_msg = {"role": "system", "content": prompts["system_prompt"]}
        user_msg = {"role": "user", "content": prompts["user_prompt"]}

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": [system_msg, user_msg],
            "modalities": ["image", "text"],
        }

        # 尺寸控制
        orientation = (prompts.get("meta", {}) or {}).get("orientation", "auto")
        size = self._get_size_for_orientation(orientation)
        payload["size"] = size

        return payload

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

            return None
        except Exception:
            return None

    # ============================================================
    # Images API (/images/generations)
    # ============================================================

    def _generate_via_images(
        self,
        prompts: Dict[str, Any],
        model_id: str,
        save_path: str,
        timeout_sec: int,
        retry: int,
    ) -> Dict[str, Any]:
        """使用 /images/generations 接口生成图像"""
        endpoint = f"{self.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        meta = prompts.get("meta", {}) or {}
        orientation = meta.get("orientation", "auto")
        size = self._get_size_for_orientation(orientation)

        # 将 system_prompt 和 user_prompt 合并为一个 prompt
        full_prompt = f"{prompts['system_prompt']}\n\n{prompts['user_prompt']}"

        payload = {
            "model": model_id,
            "prompt": full_prompt,
            "n": 1,
            "size": size,
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
                    "meta": meta,
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
            "meta": meta,
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

    def _get_size_for_orientation(self, orientation: str) -> str:
        """根据 orientation 返回尺寸"""
        if orientation == "landscape":
            return "1536x1024"
        # portrait / auto 默认竖版
        return "1024x1536"

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
# 交互模式
# ============================================================

def interactive_mode(gen: ImageGenerator, model: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    age_mode = AgeMode.KIDS
    orientation = Orientation.PORTRAIT

    print("\n儿童涂色卡生成器 v3（统一 API 架构）")
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

    if args.interactive:
        interactive_mode(gen, args.model, out_dir)
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
