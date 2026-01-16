#!/usr/bin/env python3
"""
儿童涂色卡生成器 v3.2（支持 Gemini 原生 API）
- 支持三种 API 格式：
  - /chat/completions（OpenAI 兼容格式）
  - /images/generations（GPT-Image、DALL-E）
  - /v1beta/models/{model}:generateContent（Gemini 原生格式）
- 根据模型配置自动选择 API 类型
- 支持切换不同 API 平台（通过 base_url）
- 提示词系统优化：故事优先，规则护航

环境变量：
  API_KEY=xxxx
  API_BASE_URL=https://yunwu.ai/v1

用法示例：
  python coloring_generator.py --generate "A friendly dragon" --model gpt-image-1-mini
  python coloring_generator.py --generate "机器人在球场打球" --model gemini-image --auto
  python coloring_generator.py --batch
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
    CHAT_COMPLETIONS = "chat"      # /v1/chat/completions（OpenAI 兼容格式）
    IMAGES_GENERATIONS = "images"  # /v1/images/generations（GPT-Image、DALL-E 等模型使用）
    GEMINI_NATIVE = "gemini"       # /v1beta/models/{model}:generateContent（Gemini 原生格式）


@dataclass
class IdeaInput:
    """单输入 idea 模式"""
    idea: str
    age_mode: AgeMode = AgeMode.KIDS
    orientation: Orientation = Orientation.AUTO


# ============================================================
# 模型配置
# ============================================================

DEFAULT_SIZES = {"portrait": "1024x1536", "landscape": "1536x1024"}
GEMINI_ASPECT_RATIOS = {"portrait": "2:3", "landscape": "4:3"}

MODEL_CONFIG = {
    "gpt-image-1": {"model_id": "gpt-image-1", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "OPENAI_API_KEY"},
    "gpt-image-1-mini": {"model_id": "gpt-image-1-mini", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "OPENAI_API_KEY"},
    "gpt-image-1.5": {"model_id": "gpt-image-1.5", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "OPENAI_API_KEY"},
    "gemini-image": {"model_id": "gemini-3-pro-image-preview", "api_type": APIType.GEMINI_NATIVE, "api_key_env": "GEMINI_API_KEY"},
    "default": {"model_id": "gpt-image-1", "api_type": APIType.IMAGES_GENERATIONS, "api_key_env": "API_KEY"},
}


# ============================================================
# Prompt System v2.0（故事优先，规则护航）
# ============================================================

class ColoringPromptSystem:
    """
    Prompt 架构 v2.0：
      - system_prompt：创意使命 + 故事工具 + 技术护栏
      - user_prompt：idea + age + 创意流程引导
    
    核心理念：把 AI 注意力从"避免犯错"转向"讲一个小故事"
    """

    SYSTEM_PROMPT = """You are a beloved children's picture-book illustrator creating coloring pages.
Your pages have appeared in countless nurseries and classrooms because children LOVE them.

═══════════════════════════════════════════════════════════════
YOUR CREATIVE MISSION
═══════════════════════════════════════════════════════════════
You tell tiny visual stories that make children smile, wonder, or giggle.
Every page is a "frozen moment" from an imaginary adventure — 
a scene so inviting that children can't wait to bring it to life with their crayons.

═══════════════════════════════════════════════════════════════
STORYTELLING TOOLKIT (select 2-3 per page)
═══════════════════════════════════════════════════════════════
These are your creative ingredients. Mix them to spark joy:

SCALE PLAY
  - Tiny creature in a huge world (mouse exploring a teacup forest)
  - Giant friend in a small space (friendly elephant squeezed in a bathtub)
  
GENTLE MISCHIEF  
  - A character doing something slightly silly or unexpected
  - Playful rule-breaking that feels innocent, not naughty
  
COZY MAGIC
  - Everyday objects with a touch of wonder (books that grow flowers)
  - Familiar scenes with one magical twist
  
FRIENDSHIP MOMENTS
  - Two unlikely friends sharing something
  - A helping hand, a shared umbrella, a gift being given
  
JOURNEY & DISCOVERY
  - A path leading somewhere mysterious
  - An open door, a treasure map, a distant destination
  - Someone peeking around a corner or over a hill
  
EXPRESSIVE CHARACTERS
  - Clear emotions through pose: leaning in curiosity, jumping for joy
  - Body language that tells the story without words
  - Eyes looking AT something interesting (guides viewer's attention)

═══════════════════════════════════════════════════════════════
VISUAL COMPOSITION PRINCIPLES  
═══════════════════════════════════════════════════════════════
HIERARCHY
  - One clear STAR of the scene (largest, most central, most detailed)
  - 1-2 supporting elements that SERVE the story (not random decorations)
  - Background WHISPERS the setting, doesn't compete

BREATHING ROOM
  - White space is part of the design, not emptiness to fill
  - Generous margins; no element touching the edge
  - Space between objects = easier coloring + visual clarity

DEPTH WITHOUT COMPLEXITY
  - Simple overlap: foreground partially covers background
  - Size difference: closer = bigger, farther = smaller  
  - Ground line or horizon gives spatial anchor

SILHOUETTE TEST
  - If you blur your eyes, can you still "read" the scene?
  - Main subject should be instantly recognizable as a shape

═══════════════════════════════════════════════════════════════
COLORING-FRIENDLY EXECUTION (Technical Guardrails)
═══════════════════════════════════════════════════════════════
These rules protect the child's coloring experience. 
They are constraints that ENABLE creativity, not limit it.

LINE WORK
  ✓ Pure black outlines on pure white background
  ✓ Uniform line weight throughout (consistent stroke width)
  ✓ Smooth, continuous curves; confident strokes
  ✓ ALL shapes fully closed (no gaps where color could "leak")
  
COLORABILITY  
  ✓ Every enclosed region is comfortably large enough to color
  ✓ Generous spacing between lines (small hands need room)
  ✓ Rounded, friendly forms preferred over sharp angles
  ✓ Clear distinction between coloring regions
  
ABSOLUTE RESTRICTIONS
  ✗ No shading, hatching, gradients, or gray tones
  ✗ No filled black areas (everything should be colorable)
  ✗ No text, letters, numbers, logos, or symbols
  ✗ No textures or patterns that create micro-regions
  ✗ No overlapping lines, tangles, or doubled strokes
  ✗ No tiny details smaller than a crayon tip could color

═══════════════════════════════════════════════════════════════
INTERNAL QUALITY CHECK (before finalizing)
═══════════════════════════════════════════════════════════════
Ask yourself:
  □ Is there a clear "story" a child could describe in one sentence?
  □ Would a 4-year-old know exactly what to color first?
  □ Are ALL regions closed and large enough for small hands?
  □ Does it make you smile?
"""

    AGE_MODE_PROMPTS = {
        AgeMode.TODDLER: """
╔═══════════════════════════════════════════════════════════════╗
║  AGE MODE: TODDLER & PRESCHOOL (Ages 2-4)                     ║
╚═══════════════════════════════════════════════════════════════╝

STORY STYLE: Comfort, Recognition & Gentle Wonder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Toddlers love seeing FAMILIAR things in GENTLY magical ways.
The joy comes from recognition + a small surprise.

STORY INSPIRATIONS:
  • A teddy bear having a tea party
  • A bunny wearing rain boots, splashing in puddles  
  • A kitten napping inside a cozy mitten
  • A friendly sun peeking over a hill saying good morning
  • A baby elephant giving flowers to mama elephant
  
EMOTIONAL TONE:
  • "Goodnight Moon" energy: cozy, warm, safe
  • Big friendly faces with clear, simple emotions (happy, sleepy, curious)
  • Nothing scary, chaotic, or overwhelming
  • Huggable characters they'd want as stuffed animals

VISUAL COMPLEXITY: VERY LOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 3-5 large, clearly separated objects TOTAL
  • Very large, open coloring regions (think: entire belly, entire cloud)
  • Extra-wide spacing between all elements
  • Round, soft, blob-like shapes preferred
  • Faces should be BIG with simple features (dot eyes, curved smile)
  • No interior patterns or decorative details
  • Background: minimal or none (solid ground + sky cue is enough)

SILHOUETTE RULE:
  A toddler should recognize the main character even as a solid black shape.
""",

        AgeMode.KIDS: """
╔═══════════════════════════════════════════════════════════════╗
║  AGE MODE: KIDS (Ages 5-8)                                    ║
╚═══════════════════════════════════════════════════════════════╝

STORY STYLE: Adventure, Humor & "What Happens Next?"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Kids this age love ACTION, JOKES, and IMAGINATION.
They want to see characters DOING things, not just posing.

STORY INSPIRATIONS:
  • A pirate cat sailing a paper boat in a bathtub ocean
  • A dragon who's afraid of a tiny butterfly
  • A robot learning to ride a bicycle (wobbly!)
  • A wizard's spell gone hilariously wrong (frog in the soup!)
  • An astronaut having a picnic on the moon with alien friends
  • A superhero whose cape got stuck in a door
  
NARRATIVE TECHNIQUES:
  • Mid-action freeze: character caught in the middle of doing something
  • Visual humor: mild absurdity that makes kids giggle
  • "What happens next?" tension: something about to happen
  • Role reversal: small creature being brave, big creature being shy
  • Problem-solving moment: character figuring something out

EMOTIONAL RANGE:
  • Excitement, determination, surprise, mischief, pride
  • Characters can show effort, concentration, slight worry (resolved)
  • Friendships with personality differences (brave one + shy one)

VISUAL COMPLEXITY: MODERATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • 6-12 distinct elements total (foreground + meaningful background)
  • Medium-to-large coloring regions (still no micro-details)
  • Dynamic poses allowed: running, jumping, reaching, flying
  • Simple background elements that SUPPORT the story context
  • 1-2 larger decorative elements allowed (but purposeful, not random)
  • Characters can have simple accessories (hat, tool, bag)
  • Expressions can be more nuanced (determined squint, excited grin)

BACKGROUND GUIDANCE:
  Background should answer "where is this happening?" without overwhelming.
  A few simple shapes suggesting location (trees, clouds, furniture outlines).
"""
    }

    IDEA_TO_COLORING_TEMPLATE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CREATE A DELIGHTFUL COLORING PAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE IDEA:
{idea}

{age_mode_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR CREATIVE PROCESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: FIND THE HEART
  What makes this idea FUN, SWEET, or FUNNY?
  What's the emotion you want to evoke?
  
STEP 2: CHOOSE THE MOMENT  
  Pick ONE specific instant that captures the heart.
  Not "a cat" but "a cat stretching after a nap, mid-yawn"
  Not "a rocket" but "a rocket just lifting off, with excited passengers waving"

STEP 3: ADD STORY DEPTH (not decoration)
  Include 1-2 details that DEEPEN the narrative:
    ✓ A reaction (surprised bird watching the scene)
    ✓ A cause/effect (spilled milk next to happy cat)
    ✓ An environment cue (rain clouds above the umbrella)
  Avoid random decoration that doesn't serve the story.

STEP 4: COMPOSE WITH BREATHING ROOM
  - Place your star character prominently
  - Let supporting elements orbit naturally  
  - Leave white space — it's not emptiness, it's rest for the eyes
  - Check: can a child tell what to color first?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL REQUIREMENTS (non-negotiable for print quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Pure black outlines on pure white — no fills, no gradients, no gray
• All shapes fully closed with comfortable coloring regions  
• Uniform line weight, smooth confident strokes
• No text, symbols, tiny details, or cramped areas
• Clear silhouettes readable at a glance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL CHECK: Does this page make you want to pick up crayons?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    def build_text_prompts(self, idea_input: IdeaInput) -> Dict[str, Any]:
        """构建文本提示词"""
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
- panoramic or "wide view" composition

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
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
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
    3. 提示词系统 v2.0：故事优先，规则护航
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
            key = os.getenv("API_KEY")
        if not key:
            raise ValueError(f"请设置 {env_name} 或 API_KEY 环境变量")
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
        orientation = idea_input.orientation.value

        if config["api_type"] == APIType.CHAT_COMPLETIONS:
            return self._generate_via_chat(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)
        elif config["api_type"] == APIType.GEMINI_NATIVE:
            print(f"Using Gemini Native API for model {config['model_id']}")
            return self._generate_via_gemini(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)
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
        """构建 chat/completions 请求 payload"""
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

            # 3) content 为字符串
            if isinstance(content, str):
                if content.startswith("http") or content.startswith("data:image"):
                    return content

                # 4) Markdown 格式图像
                match = re.search(r'!\[.*?\]\((data:image/[^)]+)\)', content)
                if match:
                    return match.group(1)
                match = re.search(r'!\[.*?\]\((https?://[^)]+)\)', content)
                if match:
                    return match.group(1)

            return None
        except Exception:
            return None

    # ============================================================
    # Gemini Native API (/v1beta/models/{model}:generateContent)
    # ============================================================

    def _generate_via_gemini(
        self, prompts: Dict[str, Any], model_id: str, api_key: str, orientation: str, save_path: str, timeout_sec: int, retry: int
    ) -> Dict[str, Any]:
        """使用 Gemini 原生 generateContent 接口生成图像"""
        # 构建端点: https://yunwu.ai/v1beta/models/{model}:generateContent
        base = self.base_url.replace("/v1", "")  # 移除 /v1 后缀
        # https://yunwu.ai/v1beta/models/gemini-3-pro-image-preview:generateContent
        endpoint = f"{base}/v1beta/models/{model_id}:generateContent"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = self._build_gemini_payload(prompts, orientation)

        t0 = time.time()
        last_err = None

        for attempt in range(retry + 1):
            try:
                print(f"  [Gemini] 请求端点: {endpoint}")
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
                resp.raise_for_status()
                result = resp.json()

                image_data = self._extract_image_from_gemini(result)
                if not image_data:
                    raise RuntimeError(f"Gemini 未返回图像数据: {json.dumps(result, ensure_ascii=False)[:400]}")

                self._save_image(image_data, save_path)
                self._print_saved_image_size(save_path)

                return {
                    "success": True,
                    "model": model_id,
                    "api_type": "gemini",
                    "elapsed_sec": round(time.time() - t0, 3),
                    "save_path": save_path,
                    "meta": prompts.get("meta", {}),
                    "user_prompt_preview": prompts["user_prompt"][:500],
                }

            except Exception as e:
                last_err = str(e)
                print(f"  [Gemini] 尝试 {attempt + 1} 失败: {last_err}")
                if attempt < retry:
                    time.sleep(1.2 * (attempt + 1))
                continue

        return {
            "success": False,
            "model": model_id,
            "api_type": "gemini",
            "elapsed_sec": round(time.time() - t0, 3),
            "error": last_err,
            "meta": prompts.get("meta", {}),
            "user_prompt_preview": prompts["user_prompt"][:500],
        }

    def _build_gemini_payload(self, prompts: Dict[str, Any], orientation: str) -> Dict[str, Any]:
        """构建 Gemini generateContent 请求 payload

        比例设置：
          - portrait (纵向A4) → 2:3
          - landscape (横向A4) → 4:3
        """
        combined_prompt = f"{prompts['system_prompt']}\n\n{prompts['user_prompt']}"
        aspect_ratio = GEMINI_ASPECT_RATIOS.get(orientation, "2:3")

        return {
            "contents": [
                {
                    "parts": [
                        {"text": combined_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": "2K",
                }
            }
        }

    def _extract_image_from_gemini(self, result: Dict[str, Any]) -> Optional[str]:
        """从 Gemini generateContent 响应中提取图像"""
        try:
            candidates = result.get("candidates", [])
            if not candidates:
                return None

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                # inlineData 格式（base64）
                inline_data = part.get("inlineData")
                if inline_data:
                    data = inline_data.get("data")
                    if data:
                        return data

                # fileData 格式（URL）
                file_data = part.get("fileData")
                if file_data:
                    uri = file_data.get("fileUri")
                    if uri:
                        return uri

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
    """默认测试集"""
    return [
        # 幼儿模式
        IdeaInput("A smiling teddy bear having a tea party", AgeMode.TODDLER, Orientation.AUTO),
        IdeaInput("A bunny wearing rain boots, splashing in puddles", AgeMode.TODDLER, Orientation.AUTO),
        IdeaInput("A friendly sun peeking over a hill", AgeMode.TODDLER, Orientation.PORTRAIT),
        
        # 儿童模式 - 动作场景
        IdeaInput("A pirate cat sailing a paper boat in a bathtub ocean", AgeMode.KIDS, Orientation.AUTO),
        IdeaInput("A dragon who's afraid of a tiny butterfly", AgeMode.KIDS, Orientation.AUTO),
        IdeaInput("A robot learning to ride a bicycle, looking wobbly", AgeMode.KIDS, Orientation.AUTO),
        
        # 儿童模式 - 横版场景
        IdeaInput("An astronaut having a picnic on the moon with alien friends", AgeMode.KIDS, Orientation.LANDSCAPE),
        IdeaInput("A magical underwater world with playful fish and a friendly octopus", AgeMode.KIDS, Orientation.LANDSCAPE),
    ]


def run_batch(gen: ImageGenerator, cases: List[IdeaInput], model: str, out_dir: str, retry: int) -> Dict[str, Any]:
    """批量运行测试集"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for idx, idea_input in enumerate(cases, start=1):
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"{idx:03d}_{idea_input.age_mode.value}_{idea_input.orientation.value}_{ts}"
        png_path = str(Path(out_dir) / f"{stem}.png")
        json_path = str(Path(out_dir) / f"{stem}.json")

        print(f"\n[{idx}/{len(cases)}] model={model} age={idea_input.age_mode.value} orient={idea_input.orientation.value}")
        print(f"  idea: {idea_input.idea}")

        # AUTO 模式先检测方向
        if idea_input.orientation == Orientation.AUTO:
            try:
                detector = OrientationDetector()
                detected = detector.detect(idea_input.idea)
                idea_input = IdeaInput(idea_input.idea, idea_input.age_mode, detected)
            except Exception as e:
                print(f"  [OrientationDetector] 检测失败，使用默认 PORTRAIT: {e}")
                idea_input = IdeaInput(idea_input.idea, idea_input.age_mode, Orientation.PORTRAIT)

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
    p = argparse.ArgumentParser(description="儿童涂色卡生成器 v3.1")
    p.add_argument("--model", default="gpt-image-1-mini", help="模型别名或完整 model id")
    p.add_argument("--out", default="output_v3", help="输出目录")
    p.add_argument("--retry", type=int, default=1, help="每条用例重试次数")
    p.add_argument("--base-url", default=None, help="API base URL")

    p.add_argument("--generate", type=str, default=None, help="输入 idea 生成涂色卡")
    p.add_argument("--batch", action="store_true", help="跑默认批量测试集")

    p.add_argument("--toddler", action="store_true", help="toddler 模式（2-4岁）")
    p.add_argument("--kids", action="store_true", help="kids 模式（5-8岁，默认）")
    p.add_argument("--portrait", action="store_true", help="竖版")
    p.add_argument("--landscape", action="store_true", help="横版")
    p.add_argument("--auto", action="store_true", help="自动检测横竖版")

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

    gen = ImageGenerator(base_url=args.base_url)
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.batch:
        cases = default_test_set()
        run_batch(gen, cases, args.model, out_dir, args.retry)
        return

    if args.generate:
        if orientation == Orientation.AUTO:
            try:
                detector = OrientationDetector()
                orientation = detector.detect(args.generate)
            except Exception as e:
                print(f"[OrientationDetector] 检测失败，使用默认 PORTRAIT: {e}")
                orientation = Orientation.PORTRAIT

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
    print("=" * 60)
    print("儿童涂色卡生成器 v3.1")
    print("=" * 60)
    print("\n用法示例：")
    print('  python coloring_generator.py --generate "A friendly dragon" --auto')
    print('  python coloring_generator.py --generate "小熊喝茶" --toddler')
    print('  python coloring_generator.py --generate "海盗猫的冒险" --kids --landscape')
    print("  python coloring_generator.py --batch")
    print("\n支持的模型：")
    for name, config in MODEL_CONFIG.items():
        if name != "default":
            print(f"  {name}: {config['model_id']} ({config['api_type'].value})")


if __name__ == "__main__":
    main()
