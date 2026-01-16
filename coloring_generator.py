#!/usr/bin/env python3
"""
儿童涂色卡生成器 v3.5（新增 A4 PDF 自动生成）
- 生成 PNG 后自动转换为 A4 PDF (300 DPI)
- 年龄分组优化：LITTLE_ONES (3-5岁) / YOUNG_ARTISTS (6岁+)
- Modern Storybook Style 风格（Eric Carle + Richard Scarry + Pixar）
- 支持三种 API 格式：
  - /chat/completions（OpenAI 兼容格式）
  - /images/generations（GPT-Image、DALL-E）
  - /v1beta/models/{model}:generateContent（Gemini 原生格式）

环境变量：
  API_KEY=xxxx
  API_BASE_URL=https://yunwu.ai/v1

用法示例：
  # 单张生成（自动生成 PNG + PDF）
  python coloring_generator.py --generate "A bunny eating carrots" --little --auto
  python coloring_generator.py --generate "A bunny eating carrots" --young --auto

  # 按年龄分组批量生成（推荐使用 Gemini）
  python coloring_generator.py --batch-age --model gemini-image --out output_age
  python coloring_generator.py --batch-age --age-group 3-5years --model gemini-image
  python coloring_generator.py --batch-age --age-group 5+years --model gemini-image
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
    LITTLE_ONES = "little_ones"      # 3-5岁
    YOUNG_ARTISTS = "young_artists"  # 6岁+


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
    age_mode: AgeMode = AgeMode.YOUNG_ARTISTS
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
# A4 尺寸转换（PNG + PDF）
# ============================================================

def resize_to_a4(png_path: str, dpi: int = 300) -> Optional[str]:
    """
    将 PNG 调整为 A4 尺寸（覆盖原文件）

    Args:
        png_path: PNG 文件路径
        dpi: 目标 DPI（默认 300）

    Returns:
        成功返回 png_path，失败返回 None

    A4 尺寸参考:
        210mm × 297mm
        @ 300 DPI: Portrait 2480×3508, Landscape 3508×2480
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [A4] 警告: Pillow 未安装，无法调整尺寸")
        return None

    try:
        img = Image.open(png_path)
        original_size = f"{img.width}x{img.height}"
        is_landscape = img.width > img.height

        # A4 @ 300 DPI
        if is_landscape:
            a4_w = round(297 * dpi / 25.4)  # 3508
            a4_h = round(210 * dpi / 25.4)  # 2480
        else:
            a4_w = round(210 * dpi / 25.4)  # 2480
            a4_h = round(297 * dpi / 25.4)  # 3508

        # 按比例缩放到 A4
        ratio = min(a4_w / img.width, a4_h / img.height)
        new_w = int(img.width * ratio)
        new_h = int(img.height * ratio)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 居中放置在白色 A4 背景
        a4_img = Image.new("RGB", (a4_w, a4_h), "white")
        x = (a4_w - new_w) // 2
        y = (a4_h - new_h) // 2

        # 处理透明通道
        if img_resized.mode == "RGBA":
            a4_img.paste(img_resized, (x, y), img_resized)
        else:
            if img_resized.mode != "RGB":
                img_resized = img_resized.convert("RGB")
            a4_img.paste(img_resized, (x, y))

        # 覆盖保存 PNG
        a4_img.save(png_path, "PNG")

        orientation = "landscape" if is_landscape else "portrait"
        print(f"  [A4] PNG {original_size} -> {a4_w}x{a4_h} ({orientation})")
        return png_path

    except Exception as e:
        print(f"  [A4] PNG 调整失败: {e}")
        return None


def convert_to_a4_pdf(png_path: str, pdf_path: Optional[str] = None, dpi: int = 300) -> Optional[str]:
    """
    将 PNG 转换为 A4 PDF（假设 PNG 已经是 A4 尺寸）

    Args:
        png_path: PNG 文件路径（应已是 A4 尺寸）
        pdf_path: PDF 输出路径（默认与 PNG 同名）
        dpi: 输出 DPI（默认 300）

    Returns:
        PDF 文件路径，失败返回 None
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [PDF] 警告: Pillow 未安装，无法生成 PDF")
        return None

    if pdf_path is None:
        pdf_path = png_path.rsplit(".", 1)[0] + ".pdf"

    try:
        img = Image.open(png_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 保存为 PDF
        img.save(pdf_path, "PDF", resolution=dpi)
        print(f"  [PDF] -> {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"  [PDF] 生成失败: {e}")
        return None


def process_to_a4(png_path: str, dpi: int = 300) -> Dict[str, Optional[str]]:
    """
    完整的 A4 处理流程：调整 PNG 尺寸 + 生成 PDF

    Args:
        png_path: PNG 文件路径
        dpi: 目标 DPI（默认 300）

    Returns:
        {"png_path": str|None, "pdf_path": str|None}
    """
    result = {"png_path": None, "pdf_path": None}

    # 1. 调整 PNG 到 A4 尺寸
    if resize_to_a4(png_path, dpi):
        result["png_path"] = png_path

        # 2. 生成 PDF
        pdf_path = convert_to_a4_pdf(png_path, dpi=dpi)
        if pdf_path:
            result["pdf_path"] = pdf_path

    return result


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
YOUR VISUAL STYLE: MODERN STORYBOOK
═══════════════════════════════════════════════════════════════
Your style blends the best of Western children's book illustration:

VISUAL REFERENCES (internalize these aesthetics):
  • Eric Carle: Bold simplicity, clear outlines, large colorable areas
  • Richard Scarry: Warmth, personality, gentle storytelling
  • Pixar character design: Exaggerated expressions, big friendly eyes

CORE AESTHETIC PRINCIPLES:
  ✓ ROUNDED & FRIENDLY: Soft curves, avoid sharp angles
  ✓ CHUBBY & HUGGABLE: Pleasantly plump proportions, big heads, big eyes
  ✓ CLEAR HIERARCHY: One star, supporting cast, simple background
  ✓ BREATHING ROOM: White space is design, not emptiness to fill
  ✓ EXPRESSIVE FACES: Large eyes that show emotion, simple curved smiles

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
COLORING-FRIENDLY CHECKLIST (before finalizing)
═══════════════════════════════════════════════════════════════
Ask yourself:
  □ Is there a clear "story" a child could describe in one sentence?
  □ Would a child know exactly what to color first?
  □ Are ALL regions closed and large enough for small hands?
  □ Are all shapes ROUNDED and FRIENDLY (no sharp angles)?
  □ Is the character CHUBBY and HUGGABLE looking?
  □ Does the page have enough WHITE SPACE (not crowded)?
  □ Does it make you smile?
"""

    AGE_MODE_PROMPTS = {
        AgeMode.LITTLE_ONES: """
╔═══════════════════════════════════════════════════════════════╗
║  AGE MODE: LITTLE ONES (Ages 3-5)                             ║
╚═══════════════════════════════════════════════════════════════╝

DESIGN PHILOSOPHY: "One Hero + Minimal World"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Think of it as: ONE main character that a 3-year-old could color
with just 3 crayons and still feel proud.

WESTERN STORYBOOK CHARACTER EXAMPLES:
  • Chubby teddy bear with big round belly
  • Plump bunny with floppy ears
  • Roly-poly puppy with oversized paws
  • Squishy kitten curled up in a ball
  • Round-faced duckling with tiny wings

EMOTIONAL TONE: Warm, Safe, Familiar
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • "Goodnight Moon" energy: cozy, gentle, reassuring
  • Characters look like stuffed animals they'd want to hug
  • Simple emotions: happy, sleepy, curious, surprised
  • Nothing scary, chaotic, or overwhelming

VISUAL SPECIFICATIONS (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ELEMENT COUNT:
  • MAXIMUM 3-5 large elements total on the page
  • Each coloring region must be ≥8% of page area

LINE WEIGHT:
  • BOLD, thick outlines throughout
  • Consistent heavy stroke weight

SHAPES:
  • 100% rounded, curved shapes — NO sharp angles
  • Chubby, puffy, blob-like forms
  • Think: circles, ovals, soft curves only

INTERIOR DETAILS:
  • FORBIDDEN: No patterns, textures, or internal lines
  • Faces: Simple dot eyes + curved smile only
  • Bodies: Solid colorable areas, no stripes/spots/decorations

BACKGROUND:
  • MINIMAL or NONE
  • At most: simple ground line + 1-2 tiny decorations (single flower, small cloud)
  • Background elements must be much smaller than main character

THE 3-CRAYON TEST:
  Could a 3-year-old complete this page with just 3 different crayons
  and feel satisfied? If not, simplify further.
""",

        AgeMode.YOUNG_ARTISTS: """
╔═══════════════════════════════════════════════════════════════╗
║  AGE MODE: YOUNG ARTISTS (Ages 6+)                            ║
╚═══════════════════════════════════════════════════════════════╝

DESIGN PHILOSOPHY: "Character DOING Something"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The focus shifts from "what is it?" to "what is happening?"
These kids want ACTION, HUMOR, and STORY.

NARRATIVE INSPIRATIONS:
  • A pirate cat sailing a paper boat in a bathtub ocean
  • A dragon who's terrified of a tiny butterfly
  • A robot learning to ride a bicycle (wobbling!)
  • A wizard whose spell went hilariously wrong
  • An astronaut having a picnic with alien friends
  • A superhero whose cape got stuck in a door

STORYTELLING TECHNIQUES:
  • Mid-action freeze: caught in the middle of doing something
  • Visual humor: mild absurdity that makes kids giggle
  • "What happens next?" tension
  • Role reversal: tiny creature being brave, big creature being shy
  • Problem-solving moment: character figuring something out

EMOTIONAL RANGE:
  • Excitement, determination, surprise, mischief, pride, silliness
  • Characters can show effort, concentration, playful worry
  • Dynamic expressions: determined squint, excited grin, surprised gasp

VISUAL SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━
ELEMENT COUNT:
  • 6-12 distinct elements total
  • Each coloring region must be ≥3% of page area

LINE WEIGHT:
  • Medium-weight outlines (thinner than LITTLE_ONES but still clear)
  • Consistent stroke width throughout

SHAPES:
  • 80% curved + 20% simple geometric shapes allowed
  • Still predominantly friendly and rounded
  • Dynamic poses: running, jumping, reaching, flying

INTERIOR DETAILS:
  • ALLOWED: Simple, LARGE patterns only (big stripes, large polka dots)
  • No intricate patterns or micro-textures
  • Accessories allowed: hat, tool, bag, simple clothing details

BACKGROUND:
  • Simple scene elements that support the story context
  • Trees, clouds, furniture outlines, simple landscape shapes
  • Background should answer "where is this?" without overwhelming
  • Main character still clearly dominant in visual hierarchy

ACTION TEST:
  Can a child describe this page starting with a VERB?
  "The dragon is running from..." "The robot is learning to..." "The cat is sailing..."
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

STEP 5: AGE-APPROPRIATENESS CHECK
  Before finalizing, verify your design matches the target age:
  □ Element count within specified range?
  □ Coloring regions large enough? (check minimum % requirement)
  □ Line weight appropriate? (bold for younger, medium for older)
  □ Shape roundness matches age spec?
  □ Interior detail level appropriate?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL REQUIREMENTS (non-negotiable for print quality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Pure black outlines on pure white — no fills, no gradients, no gray
• All shapes fully closed with comfortable coloring regions
• Uniform line weight, smooth confident strokes
• No text, symbols, tiny details, or cramped areas
• Clear silhouettes readable at a glance
• Characters should look CHUBBY, ROUNDED, and HUGGABLE

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
            print(f"[OrientationDetector] LLM answer: {answer}")
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
            result = self._generate_via_chat(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)
        elif config["api_type"] == APIType.GEMINI_NATIVE:
            print(f"Using Gemini Native API for model {config['model_id']}")
            result = self._generate_via_gemini(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)
        else:
            result = self._generate_via_images(prompts, config["model_id"], api_key, orientation, save_path, timeout_sec, retry)

        # 成功后自动调整为 A4 尺寸（PNG + PDF）
        if result.get("success") and result.get("save_path"):
            a4_result = process_to_a4(result["save_path"])
            if a4_result.get("pdf_path"):
                result["pdf_path"] = a4_result["pdf_path"]

        return result

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
        # LITTLE_ONES 模式 (3-5岁)
        IdeaInput("A chubby teddy bear having a tea party", AgeMode.LITTLE_ONES, Orientation.AUTO),
        IdeaInput("A plump bunny wearing rain boots, splashing in puddles", AgeMode.LITTLE_ONES, Orientation.AUTO),
        IdeaInput("A friendly sun peeking over a hill", AgeMode.LITTLE_ONES, Orientation.PORTRAIT),

        # YOUNG_ARTISTS 模式 (6岁+) - 动作场景
        IdeaInput("A pirate cat sailing a paper boat in a bathtub ocean", AgeMode.YOUNG_ARTISTS, Orientation.AUTO),
        IdeaInput("A dragon who's terrified of a tiny butterfly", AgeMode.YOUNG_ARTISTS, Orientation.AUTO),
        IdeaInput("A robot learning to ride a bicycle, wobbling hilariously", AgeMode.YOUNG_ARTISTS, Orientation.AUTO),

        # YOUNG_ARTISTS 模式 (6岁+) - 横版场景
        IdeaInput("An astronaut having a picnic on the moon with alien friends", AgeMode.YOUNG_ARTISTS, Orientation.LANDSCAPE),
        IdeaInput("A magical underwater world with playful fish and a friendly octopus", AgeMode.YOUNG_ARTISTS, Orientation.LANDSCAPE),
    ]


# ============================================================
# 年龄分组提示词集合
# ============================================================

AGE_GROUP_PROMPTS = {
    "3-5years": [
        "A smiling dinosaur wearing a chef hat, baking one giant cookie in a tiny kitchen.",
        "A fluffy bunny driving a toy car through bubble clouds with three big balloons.",
        "A friendly whale floating in the sky like a blimp, carrying a tiny house on its back.",
        "A kitten astronaut standing on the moon holding a star-shaped flag, with one big planet behind.",
        "A happy turtle with a small flower garden growing on its shell.",
        "A tiny dragon sleeping on a pillow made of marshmallows under a big moon and a few stars.",
        "A bear riding a giant pencil like a rocket, leaving a simple rainbow trail.",
        "A duck wearing rain boots jumping into one big puddle with cheerful splashes.",
        "A smiling sun giving an ice cream to a little cloud friend.",
        "A unicorn brushing its mane in a cozy room with a bed and one window.",
        "A little elephant watering one giant flower with a small watering can.",
        "A friendly robot hugging a teddy bear with a big happy face.",
        "A strawberry-shaped house with a door and two windows, a path, and two butterflies.",
        "A penguin having a tea party with two cupcakes on a big round table.",
        "A lion with a fluffy cloud-like mane standing on grass near one big tree.",
        "A snail carrying a tiny castle on its shell, smiling proudly.",
        "A cheerful octopus juggling four big beach balls in the center.",
        "A giraffe wearing a scarf looking at one big butterfly hovering nearby.",
        "A tiny fairy riding a ladybug above one big flower.",
        "A friendly shark playing a ukulele underwater next to one coral and one fish.",
    ],
    "5+years": [
        "A hot-air balloon shaped like a giant teapot floating over a whimsical town.",
        "A dragon librarian in a grand library organizing flying books and scrolls.",
        "An underwater coral city with tiny submarine buses and fish-shaped traffic lights, a diver waving.",
        "A giant treehouse village connected by rope bridges, squirrels delivering mail between houses.",
        "A moon carnival where astronauts ride a Ferris wheel made of stars, with rocket booths nearby.",
        "A cat-run bakery where donuts float like planets around a cosmic oven, sprinkles falling like meteors.",
        "A friendly monster school classroom with backpacks, a chalkboard, and silly science experiments.",
        "A magical train traveling through a waterfall tunnel with glowing lanterns and animal-shaped luggage.",
        "A floating island farm with windmills, cloud sheep, and a rainbow river winding through fields.",
        "A robot gardener growing geometric flowers in a greenhouse, with tiny watering drones.",
        "A castle kitchen where pots and pans dance while a young wizard stirs starry soup.",
        "A submarine treasure hunt with a map, an ancient shipwreck, and friendly sea creatures guiding the way.",
        "A snowy mountain ski resort run by penguins, with lifts, cabins, and a hot cocoa stand.",
        "A candy jungle with lollipop trees and chocolate rocks, explorers crossing a syrup river on wafer rafts.",
        "A space aquarium with glass tunnels, giant jellyfish floating above, and a family of aliens visiting.",
        "A medieval festival where knights ride giant snails, with banners, stalls, and a playful crowd cheering.",
        "A wizard's workshop filled with potion bottles, gears, a talking clock, and a tiny dragon assistant.",
        "A sky harbor where airships dock, suitcases roll by themselves, and clouds look like animals.",
        "A deep-sea neighborhood where a giant octopus mail carrier delivers letters to coral houses.",
        "A fantasy city street where shadows become playful pet animals, kids chasing their shadow pets under glowing lanterns.",
    ],
}


def run_batch_by_age_groups(
    gen: ImageGenerator,
    model: str,
    out_dir: str,
    retry: int,
    age_groups: Optional[List[str]] = None,
    start_index: int = 1,
) -> Dict[str, Any]:
    """
    按年龄分组批量生成涂色卡

    Args:
        gen: ImageGenerator 实例
        model: 模型名称
        out_dir: 输出根目录
        retry: 重试次数
        age_groups: 要生成的年龄组列表，默认全部
        start_index: 起始序号（用于续传）

    Output Structure:
        out_dir/
        ├── 3-5years/
        │   ├── 001_dinosaur_chef_20260116_120000.png
        │   ├── 001_dinosaur_chef_20260116_120000.json
        │   └── ...
        ├── 5+years/
        │   ├── 001_teapot_balloon_20260116_120100.png
        │   └── ...
        └── _summary.json
    """
    if age_groups is None:
        age_groups = list(AGE_GROUP_PROMPTS.keys())

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, List[Dict[str, Any]]] = {}
    total_count = 0
    success_count = 0
    failed_count = 0

    for age_group in age_groups:
        if age_group not in AGE_GROUP_PROMPTS:
            print(f"[Warning] Unknown age group: {age_group}, skipping...")
            continue

        prompts = AGE_GROUP_PROMPTS[age_group]
        age_mode = AgeMode.LITTLE_ONES if age_group == "3-5years" else AgeMode.YOUNG_ARTISTS

        # 创建年龄组子目录
        group_dir = Path(out_dir) / age_group
        group_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        print(f"\n{'='*60}")
        print(f"Age Group: {age_group} ({len(prompts)} prompts)")
        print(f"Age Mode: {age_mode.value}")
        print(f"Output: {group_dir}")
        print(f"{'='*60}")

        for idx, prompt in enumerate(prompts, start=start_index):
            total_count += 1
            ts = time.strftime("%Y%m%d_%H%M%S")

            # 生成简短文件名（取 prompt 前几个单词）
            words = prompt.split()[:3]
            slug = "_".join(w.lower().strip(".,!?") for w in words if w.isalnum() or w.replace("-", "").isalnum())[:30]
            stem = f"{idx:03d}_{slug}_{ts}"
            png_path = str(group_dir / f"{stem}.png")
            json_path = str(group_dir / f"{stem}.json")

            print(f"\n[{age_group}] [{idx}/{len(prompts)}] model={model}")
            print(f"  prompt: {prompt[:80]}...")

            # 创建 IdeaInput，使用 AUTO 方向自动检测
            idea_input = IdeaInput(prompt, age_mode, Orientation.AUTO)

            # AUTO 模式先检测方向
            try:
                detector = OrientationDetector()
                detected = detector.detect(prompt)
                idea_input = IdeaInput(prompt, age_mode, detected)
            except Exception as e:
                print(f"  [OrientationDetector] 检测失败，使用默认 PORTRAIT: {e}")
                idea_input = IdeaInput(prompt, age_mode, Orientation.PORTRAIT)

            res = gen.generate_from_idea(
                idea_input=idea_input,
                model=model,
                save_path=png_path,
                retry=retry,
            )

            # 添加额外元数据
            res["age_group"] = age_group
            res["prompt_index"] = idx
            res["original_prompt"] = prompt

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            if res["success"]:
                success_count += 1
                pdf_info = f" + PDF" if res.get("pdf_path") else ""
                print(f"  ✓ OK   -> {res['save_path']}{pdf_info} ({res['elapsed_sec']}s)")
            else:
                failed_count += 1
                print(f"  ✗ FAIL -> {res.get('error', 'Unknown error')} ({res['elapsed_sec']}s)")

            results.append(res)

        all_results[age_group] = results

        # 保存年龄组小结
        group_summary = {
            "age_group": age_group,
            "total": len(results),
            "success": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "model": model,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(str(group_dir / "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(group_summary, f, ensure_ascii=False, indent=2)

    # 保存总汇总
    summary = {
        "total": total_count,
        "success": success_count,
        "failed": failed_count,
        "model": model,
        "out_dir": out_dir,
        "age_groups": {
            ag: {
                "total": len(all_results.get(ag, [])),
                "success": sum(1 for r in all_results.get(ag, []) if r.get("success")),
                "failed": sum(1 for r in all_results.get(ag, []) if not r.get("success")),
            }
            for ag in age_groups if ag in all_results
        },
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(str(Path(out_dir) / "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("BATCH GENERATION COMPLETE")
    print("="*60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


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
            pdf_info = f" + PDF" if res.get("pdf_path") else ""
            print(f"  ✓ OK   -> {res['save_path']}{pdf_info} ({res['elapsed_sec']}s)")
        else:
            print(f"  ✗ FAIL -> {res['error']} ({res['elapsed_sec']}s)")
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
    p = argparse.ArgumentParser(description="儿童涂色卡生成器 v3.5 - Modern Storybook Style")
    p.add_argument("--model", default="gpt-image-1-mini", help="模型别名或完整 model id")
    p.add_argument("--out", default="output_v3", help="输出目录")
    p.add_argument("--retry", type=int, default=1, help="每条用例重试次数")
    p.add_argument("--base-url", default=None, help="API base URL")

    p.add_argument("--generate", type=str, default=None, help="输入 idea 生成涂色卡")
    p.add_argument("--batch", action="store_true", help="跑默认批量测试集")

    # 按年龄分组批量生成
    p.add_argument("--batch-age", action="store_true", help="按年龄分组批量生成（3-5岁、5+岁）")
    p.add_argument("--age-group", type=str, default=None,
                   help="指定年龄组: 3-5years, 5+years (可用逗号分隔多个，默认全部)")
    p.add_argument("--start-index", type=int, default=1, help="起始序号（用于续传）")

    # 年龄模式：新参数
    p.add_argument("--little", action="store_true", help="3-5岁模式 (little_ones)")
    p.add_argument("--young", action="store_true", help="6岁+模式 (young_artists，默认)")
    # 年龄模式：旧参数别名（兼容）
    p.add_argument("--toddler", action="store_true", help="[别名] 等同于 --little")
    p.add_argument("--kids", action="store_true", help="[别名] 等同于 --young")

    p.add_argument("--portrait", action="store_true", help="竖版")
    p.add_argument("--landscape", action="store_true", help="横版")
    p.add_argument("--auto", action="store_true", help="自动检测横竖版")

    return p.parse_args()


def main():
    args = parse_args()

    # 年龄模式处理：--little/--toddler → LITTLE_ONES, --young/--kids → YOUNG_ARTISTS
    # 默认为 YOUNG_ARTISTS
    age_mode = AgeMode.YOUNG_ARTISTS
    if args.little or args.toddler:
        age_mode = AgeMode.LITTLE_ONES
    elif args.young or args.kids:
        age_mode = AgeMode.YOUNG_ARTISTS

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

    # 按年龄分组批量生成
    if args.batch_age:
        age_groups = None
        if args.age_group:
            age_groups = [g.strip() for g in args.age_group.split(",")]
        run_batch_by_age_groups(
            gen=gen,
            model=args.model,
            out_dir=out_dir,
            retry=args.retry,
            age_groups=age_groups,
            start_index=args.start_index,
        )
        return

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
    print("儿童涂色卡生成器 v3.5")
    print("=" * 60)
    print("\n年龄模式：")
    print("  --little  3-5岁模式 (little_ones): 极简、大区域、粗线条")
    print("  --young   6岁+模式 (young_artists，默认): 动态、故事感、适度细节")
    print("\n用法示例：")
    print('  python coloring_generator.py --generate "A bunny eating carrots" --little --auto')
    print('  python coloring_generator.py --generate "A bunny eating carrots" --young --auto')
    print('  python coloring_generator.py --generate "A friendly dragon" --auto')
    print('  python coloring_generator.py --generate "海盗猫的冒险" --young --landscape')
    print("  python coloring_generator.py --batch")
    print("\n按年龄分组批量生成：")
    print("  python coloring_generator.py --batch-age --model gemini-image --out output_age")
    print("  python coloring_generator.py --batch-age --age-group 3-5years --model gemini-image")
    print("  python coloring_generator.py --batch-age --age-group 5+years --model gemini-image")
    print("  python coloring_generator.py --batch-age --age-group 3-5years,5+years --start-index 5")
    print("\n可用年龄组：")
    for group, prompts in AGE_GROUP_PROMPTS.items():
        print(f"  {group}: {len(prompts)} 个提示词")
    print("\n支持的模型：")
    for name, config in MODEL_CONFIG.items():
        if name != "default":
            print(f"  {name}: {config['model_id']} ({config['api_type'].value})")


if __name__ == "__main__":
    main()
