#!/usr/bin/env python3
"""
儿童涂色卡生成器 v2（推荐方案一：单输入 idea）
- OpenRouter chat/completions
- system / user 分离 messages（更稳）
- 支持：文生图（idea->image）、图生图（image->coloring）
- 去掉：PNG->PDF / 矢量化 / reportlab / cairosvg / potrace 等后处理
- 输出：PNG + JSON 记录

环境变量：
  OPENROUTER_API_KEY=xxxx

用法示例：
  python coloring_generator_v2.py --batch --model gpt5-image-mini --out output_batch
  python coloring_generator_v2.py --generate "A friendly dragon flying over a castle" --kids --portrait --model nano-banana-pro
  python coloring_generator_v2.py --image2color ./input.jpg --kids --portrait --model gpt5-image-mini
  python coloring_generator_v2.py --interactive
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

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENROUTER_API_KEY 环境变量或传入 api_key 参数")
        self.base_url = base_url
        self.prompt_system = ColoringPromptSystem()

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
        }

        # 尺寸策略：按模型原生参数（保持你之前策略）
        if "gemini" in model_id.lower():
            payload["image_config"] = {"aspect_ratio": "2:3", "image_size": "2K"}
        else:
            payload["size"] = "1024x1536"
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
        try:
            choices = result.get("choices", [])
            if not choices:
                return None
            msg = choices[0].get("message", {})

            images = msg.get("images", [])
            if images:
                first = images[0]
                if isinstance(first, dict):
                    url = first.get("image_url", {}).get("url")
                    if url:
                        return url
                elif isinstance(first, str):
                    return first

            content = msg.get("content", "")
            if isinstance(content, str) and (content.startswith("http") or content.startswith("data:image")):
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
    p.add_argument("--model", default="gpt5-image-mini", help="模型别名或完整 model id")
    p.add_argument("--out", default="output_v2", help="输出目录")
    p.add_argument("--retry", type=int, default=1, help="每条用例重试次数")

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

    gen = OpenRouterImageGenerator()

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
    print("  python coloring_generator_v2.py --interactive")


if __name__ == "__main__":
    main()