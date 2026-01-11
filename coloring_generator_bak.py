#!/usr/bin/env python3
"""
儿童涂色卡生成器 - 通过 OpenRouter 调用文生图 API
专业提示词系统版本

输入格式: 物体 在 某某地方 做什么
变量控制:
1. Subject (谁/什么)
2. Location (在哪里)
3. Action (做什么)
4. Age Mode (≤5 / ≥6)
"""

import os
import json
import base64
import requests
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    # 从当前目录和脚本目录查找 .env
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent / ".env"
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    # 如果没有安装 python-dotenv，手动解析 .env
    def load_env_manual():
        env_paths = [
            Path.cwd() / ".env",
            Path(__file__).parent / ".env"
        ]
        for env_path in env_paths:
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key not in os.environ:
                                os.environ[key] = value
                break
    load_env_manual()


class AgeMode(Enum):
    """年龄模式"""
    TODDLER = "toddler"  # 5岁以下
    KIDS = "kids"        # 6岁及以上


class Orientation(Enum):
    """页面方向"""
    PORTRAIT = "portrait"    # 竖版
    LANDSCAPE = "landscape"  # 横版
    AUTO = "auto"            # 自动选择


@dataclass
class SceneInput:
    """场景输入结构 - 5个核心变量"""
    subject: str                                    # 物体/主体 (谁/什么)
    location: str                                   # 地点 (在哪里)
    action: str                                     # 动作 (做什么)
    age_mode: AgeMode = AgeMode.KIDS                # 年龄模式
    orientation: Orientation = Orientation.AUTO     # 页面方向


class ColoringPromptSystem:
    """
    专业儿童涂色卡提示词系统

    架构:
    SYSTEM PROMPT  → 固定
    USER PROMPT    → Text 或 Image 模板 + 年龄模式

    只控制 4 个变量:
    1. Subject（谁 / 什么）
    2. Location（在哪里）
    3. Action（做什么）
    4. Age Mode（≤5 / ≥6）
    """

    # ================================================================
    # 系统提示词 - 固定不变
    # ================================================================
    SYSTEM_PROMPT = """
        You are a professional children's coloring book illustrator
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
        - Overall style should resemble high-quality professional
        children's coloring books

        FINAL SELF-CHECK BEFORE OUTPUT:
        - Every outline is closed and continuous
        - No dark or filled regions appear
        - Line weight is consistent and print-friendly
        - Complexity matches the target age group
        - The page is ready for direct printing and enjoyable coloring
            
    """

    # ================================================================
    # 年龄模式提示词 (使用相对描述，不含物理单位)
    # ================================================================
    AGE_MODE_PROMPTS = {
        AgeMode.TODDLER: """
            [AGE MODE: TODDLER / PRESCHOOL]

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

        AgeMode.KIDS: """
            [AGE MODE: KIDS]

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

    # 方向提示词 - 强调充分利用页面空间
    ORIENTATION_PROMPTS = {
        "portrait": """VERTICAL COMPOSITION:
            - Tall, storybook-style layout
            - Sky elements near the top
            - Ground elements near the bottom
            - Let the scene flow naturally from top to bottom""",
                    "landscape": """HORIZONTAL COMPOSITION:
            - Wide, panoramic layout
            - Spread elements evenly from left to right
            - Avoid crowded or compressed areas""",
                    "auto": """AUTO COMPOSITION:
            - Naturally fill the entire page
            - Extend the scene with simple environment elements
            - No empty corners, no clutter"""
    }

    TEXT_TO_COLORING_TEMPLATE = """
    Create a children's coloring page.

    Subject: {subject}
    Location: {location}
    Action: {action}

    {age_mode_prompt}

    COMPOSITION:
    {orientation_prompt}

    The illustration should feel joyful, imaginative,
    and visually pleasing for children.

    Follow all coloring book rules from the system prompt.
    Ensure the result is print-ready and easy to color.

"""
    IMAGE_TO_COLORING_TEMPLATE = """CConvert the provided image into a children's coloring book page.

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

ARTISTIC DIRECTION:
- Cheerful, playful, storybook-like mood
- Friendly characters with simple expressions
- Whimsical feeling without visual clutter
- Add gentle imaginative elements only if they remain simple and open

FINAL REQUIREMENTS:
- Black outlines on a pure white background only
- No filled black areas, no shading, no grayscale
- No text, letters, numbers, logos, or symbols
- The result must be print-ready and safe for children to color"""

    def __init__(self):
        pass

    def get_system_prompt(self) -> str:
        """获取系统提示词（固定）"""
        return self.SYSTEM_PROMPT

    def build_text_to_coloring_prompt(self, scene: SceneInput) -> str:
        """
        构建文本生成涂色卡的用户提示词

        Args:
            scene: 包含 subject, location, action, age_mode, orientation 的场景输入

        Returns:
            str: 完整的用户提示词
        """
        # 获取年龄模式提示词
        age_mode_prompt = self.AGE_MODE_PROMPTS[scene.age_mode]

        # 获取方向提示词
        orientation_prompt = self.ORIENTATION_PROMPTS[scene.orientation.value]

        # 填充模板 (直接使用原始输入)
        user_prompt = self.TEXT_TO_COLORING_TEMPLATE.format(
            subject=scene.subject,
            location=scene.location,
            action=scene.action,
            age_mode_prompt=age_mode_prompt,
            orientation_prompt=orientation_prompt
        )

        return user_prompt

    def build_image_to_coloring_prompt(
        self,
        age_mode: AgeMode = AgeMode.KIDS,
        orientation: Orientation = Orientation.AUTO
    ) -> str:
        """
        构建图片转涂色卡的用户提示词

        Args:
            age_mode: 年龄模式
            orientation: 页面方向

        Returns:
            str: 完整的用户提示词
        """
        age_mode_prompt = self.AGE_MODE_PROMPTS[age_mode]
        orientation_prompt = self.ORIENTATION_PROMPTS[orientation.value]

        user_prompt = self.IMAGE_TO_COLORING_TEMPLATE.format(
            age_mode_prompt=age_mode_prompt,
            orientation_prompt=orientation_prompt
        )

        return user_prompt

    def get_full_prompts(self, scene: SceneInput, mode: str = "text") -> dict:
        """
        获取完整的提示词组合

        Args:
            scene: 场景输入
            mode: "text" 文生图 或 "image" 图生图

        Returns:
            dict: 包含 system_prompt 和 user_prompt
        """
        system_prompt = self.get_system_prompt()

        if mode == "text":
            user_prompt = self.build_text_to_coloring_prompt(scene)
        else:
            user_prompt = self.build_image_to_coloring_prompt(scene.age_mode, scene.orientation)

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "age_mode": scene.age_mode.value,
            "orientation": scene.orientation.value,
            "original_input": {
                "subject": scene.subject,
                "location": scene.location,
                "action": scene.action
            }
        }


class OpenRouterImageGenerator:
    """
    OpenRouter 图像生成器
    支持多种文生图模型
    """

    # OpenRouter 支持的图像生成模型
    # 通过 API 查询获取: GET https://openrouter.ai/api/v1/models
    SUPPORTED_MODELS = {
        # OpenAI GPT 图像系列 (OpenRouter 不支持自定义尺寸，固定 1024x1024)
        "gpt5-image": "openai/gpt-5-image",
        "gpt5-image-mini": "openai/gpt-5-image-mini",

        # Google Gemini 图像系列 (支持 image_config 自定义分辨率: 1K/2K/4K)
        "gemini-image": "google/gemini-2.5-flash-image",          # Nano Banana, 支持 2K
        "nano-banana-pro": "google/gemini-3-pro-image-preview",   # Nano Banana Pro

        # 默认推荐 (Gemini 支持更高分辨率)
        "default": "google/gemini-2.5-flash-image",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENROUTER_API_KEY 环境变量或传入 api_key 参数")

        self.base_url = "https://openrouter.ai/api/v1"
        self.prompt_system = ColoringPromptSystem()

    # A4 尺寸 @ 175 DPI (像素)
    A4_PORTRAIT_WIDTH = 2480
    A4_PORTRAIT_HEIGHT = 3508
    A4_LANDSCAPE_WIDTH = 3508
    A4_LANDSCAPE_HEIGHT = 2480

    def generate_from_text(
        self,
        scene: SceneInput,
        model: str = "gemini-image",
        width: Optional[int] = None,
        height: Optional[int] = None,
        save_path: Optional[str] = None,
        auto_pdf: bool = True
    ) -> dict:
        """
        文本生成涂色卡

        Args:
            scene: 场景输入 (subject, location, action, age_mode, orientation)
            model: 模型名称
            width: 图像宽度 (None = 根据 orientation 自动设置 A4 尺寸)
            height: 图像高度 (None = 根据 orientation 自动设置 A4 尺寸)
            save_path: 保存路径
            auto_pdf: 是否自动生成 PDF
        """
        # 根据 orientation 确定尺寸
        if width is None or height is None:
            if scene.orientation == Orientation.LANDSCAPE:
                width = self.A4_LANDSCAPE_WIDTH
                height = self.A4_LANDSCAPE_HEIGHT
            else:  # PORTRAIT 或 AUTO 默认使用竖版
                width = self.A4_PORTRAIT_WIDTH
                height = self.A4_PORTRAIT_HEIGHT

        prompts = self.prompt_system.get_full_prompts(scene, mode="text")
        return self._generate(prompts, model, width, height, save_path, auto_pdf=auto_pdf)

    def generate_from_image(
        self,
        image_path: str,
        age_mode: AgeMode = AgeMode.KIDS,
        orientation: Orientation = Orientation.AUTO,
        model: str = "gpt5-image",
        width: Optional[int] = None,
        height: Optional[int] = None,
        save_path: Optional[str] = None,
        auto_pdf: bool = True
    ) -> dict:
        """
        图片转涂色卡

        Args:
            image_path: 源图片路径
            age_mode: 年龄模式
            orientation: 页面方向
            model: 模型名称
            width: 输出宽度 (None = 根据 orientation 自动设置 A4 尺寸)
            height: 输出高度 (None = 根据 orientation 自动设置 A4 尺寸)
            save_path: 保存路径
            auto_pdf: 是否自动生成 PDF
        """
        # 根据 orientation 确定尺寸
        if width is None or height is None:
            if orientation == Orientation.LANDSCAPE:
                width = self.A4_LANDSCAPE_WIDTH
                height = self.A4_LANDSCAPE_HEIGHT
            else:  # PORTRAIT 或 AUTO 默认使用竖版
                width = self.A4_PORTRAIT_WIDTH
                height = self.A4_PORTRAIT_HEIGHT

        scene = SceneInput(subject="", location="", action="", age_mode=age_mode, orientation=orientation)
        prompts = self.prompt_system.get_full_prompts(scene, mode="image")
        prompts["source_image"] = image_path

        return self._generate(prompts, model, width, height, save_path, image_path, auto_pdf=auto_pdf)

    def _generate(
        self,
        prompts: dict,
        model: str,
        width: int,
        height: int,
        save_path: Optional[str],
        image_path: Optional[str] = None,
        auto_pdf: bool = True
    ) -> dict:
        """执行图像生成"""
        model_id = self.SUPPORTED_MODELS.get(model, model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://coloring-generator.local",
            "X-Title": "Children Coloring Page Generator"
        }

        # OpenRouter 所有模型都通过 chat/completions endpoint
        endpoint = f"{self.base_url}/chat/completions"
        payload = self._build_payload(prompts, model_id, width, height, image_path)

        self._print_generation_info(prompts, model_id, width, height)

        try:
            # 显示关键参数（不显示完整 prompt）
            payload_preview = {k: v for k, v in payload.items() if k != "messages"}
            payload_preview["messages"] = f"[{len(payload.get('messages', []))} messages]"
            print("payload:", json.dumps(payload_preview, ensure_ascii=False))
            response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()

            image_data = self._extract_image(result, model_id)

            pdf_path = None
            if save_path and image_data:
                self._save_image(image_data, save_path, target_width=width, target_height=height)
                print(f"\n图像已保存至: {save_path}")

                # 自动生成 PDF
                if auto_pdf:
                    try:
                        processor = ImagePostProcessor()
                        pdf_path = processor.png_to_pdf(save_path, vectorize=True)
                        if pdf_path:
                            print(f"PDF 已生成: {pdf_path}")
                    except Exception as e:
                        print(f"PDF 生成失败: {e}")

            return {
                "success": True,
                "prompts": prompts,
                "model": model_id,
                "image_data": image_data,
                "save_path": save_path,
                "pdf_path": pdf_path,
                "raw_response": result
            }

        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            # 尝试获取更详细的错误信息
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.json()
                    error_msg = f"{error_msg}\n详细: {json.dumps(error_detail, ensure_ascii=False)}"
            except:
                pass
            return {
                "success": False,
                "error": error_msg,
                "prompts": prompts,
                "model": model_id
            }

    def _build_payload(
        self,
        prompts: dict,
        model_id: str,
        width: int,
        height: int,
        image_path: Optional[str] = None
    ) -> dict:
        """构建 OpenRouter API 请求体"""

        # 合并 system prompt 和 user prompt 作为完整提示
        full_prompt = f"{prompts['system_prompt']}\n\n{prompts['user_prompt']}"

        # 构建消息
        messages = [
            {"role": "user", "content": full_prompt}
        ]

        # 如果是图生图，添加图片
        if image_path:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            # 获取图片类型
            ext = Path(image_path).suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }.get(ext, "image/png")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": full_prompt
                        }
                    ]
                }
            ]

        payload = {
            "model": model_id,
            "messages": messages,
        }

        # 根据模型类型添加特定参数
        # A4 比例: 210/297 = 0.707
        # 最接近的标准比例: 2:3 = 0.667 (竖版), 3:2 = 1.5 (横版)

        if "gemini" in model_id.lower():
            # Gemini 模型支持 image_config 参数
            # aspect_ratio: "1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9" 等
            # image_size: "1K", "2K", "4K" (必须大写!)
            if height > width:
                aspect = "2:3"   # 竖版，最接近 A4 (差距 4.0%)
            elif width > height:
                aspect = "3:2"   # 横版
            else:
                aspect = "1:1"   # 正方形
            payload["image_config"] = {
                "aspect_ratio": aspect,
                "image_size": "2K",  # 更高分辨率
            }

        elif "openai" in model_id.lower() or "gpt" in model_id.lower():
            # OpenAI GPT Image 模型
            # 支持: 1024x1024, 1024x1536 (竖版), 1536x1024 (横版)
            # 1024x1536 最接近 A4 @ 150DPI (1240x1754)
            if height > width:
                size = "1024x1536"  # 竖版 (比例 0.667)
            elif width > height:
                size = "1536x1024"  # 横版 (比例 1.5)
            else:
                size = "1024x1024"  # 正方形
            payload["size"] = size
            payload["quality"] = "high"

        elif "flux" in model_id.lower():
            # Flux 模型的图像参数
            payload["provider"] = {
                "sort": "throughput"
            }

        elif "dall-e" in model_id.lower():
            # DALL-E 特定参数
            if height > width:
                size = "1024x1792"
            elif width > height:
                size = "1792x1024"
            else:
                size = "1024x1024"
            payload["size"] = size
            payload["quality"] = "standard"

        return payload

    def _extract_image(self, result: dict, model_id: str) -> Optional[str]:
        """提取图像数据"""
        try:
            # OpenRouter 返回格式
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})

                # Gemini 图像模型: 图像在 message.images 数组中
                images = message.get("images", [])
                if images:
                    first_image = images[0]
                    if isinstance(first_image, dict):
                        image_url = first_image.get("image_url", {})
                        url = image_url.get("url", "")
                        if url:
                            return url
                    elif isinstance(first_image, str):
                        return first_image

                # 检查 content 字段
                content = message.get("content", "")

                # 检查是否是 URL
                if content and content.startswith("http"):
                    return content

                # 检查是否包含 markdown 图片链接
                if content:
                    import re
                    url_match = re.search(r'!\[.*?\]\((https?://[^\)]+)\)', content)
                    if url_match:
                        return url_match.group(1)

                    # 检查是否是 base64 data URL
                    if content.startswith("data:image"):
                        return content

            # 旧格式兼容
            data = result.get("data", [{}])
            if data:
                return data[0].get("b64_json") or data[0].get("url")

        except (IndexError, KeyError) as e:
            print(f"提取图像失败: {e}")
            return None

    def _save_image(self, image_data: str, save_path: str, target_width: int = 0, target_height: int = 0, crop_to_fill: bool = False):
        """
        保存图像，可选缩放到目标尺寸

        Args:
            crop_to_fill: True=裁剪填充(无白边), False=缩放填充(保留白边)
        """
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # 先获取原始图像数据
        if image_data.startswith("http"):
            # URL 格式
            response = requests.get(image_data, timeout=60)
            raw_data = response.content
        elif image_data.startswith("data:image"):
            # data URL 格式: data:image/png;base64,XXXX
            if "," in image_data:
                base64_data = image_data.split(",", 1)[1]
            else:
                base64_data = image_data
            raw_data = base64.b64decode(base64_data)
        else:
            # 纯 base64 格式
            raw_data = base64.b64decode(image_data)

        # 如果指定了目标尺寸，进行缩放
        if target_width > 0 and target_height > 0:
            try:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(raw_data))
                original_size = f"{img.width}x{img.height}"

                img_ratio = img.width / img.height
                target_ratio = target_width / target_height

                if crop_to_fill:
                    # 裁剪填充模式: 放大后居中裁剪，无白边
                    if img_ratio > target_ratio:
                        new_height = target_height
                        new_width = int(target_height * img_ratio)
                    else:
                        new_width = target_width
                        new_height = int(target_width / img_ratio)

                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # 居中裁剪
                    left = (new_width - target_width) // 2
                    top = (new_height - target_height) // 2
                    right = left + target_width
                    bottom = top + target_height

                    cropped = img_resized.crop((left, top, right, bottom))

                    # 创建白色背景，处理透明通道
                    final_img = Image.new("RGB", (target_width, target_height), "white")
                    if cropped.mode == "RGBA":
                        final_img.paste(cropped, (0, 0), cropped)
                    else:
                        final_img.paste(cropped, (0, 0))

                    final_img.save(save_path, "PNG")
                    print(f"图像已裁剪填充: {original_size} → {target_width}x{target_height}")
                else:
                    # 缩放填充模式: 保持比例，白色背景填充
                    if img_ratio > target_ratio:
                        new_width = target_width
                        new_height = int(target_width / img_ratio)
                    else:
                        new_height = target_height
                        new_width = int(target_height * img_ratio)

                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    final_img = Image.new("RGB", (target_width, target_height), "white")
                    x = (target_width - new_width) // 2
                    y = (target_height - new_height) // 2

                    if img_resized.mode == "RGBA":
                        final_img.paste(img_resized, (x, y), img_resized)
                    else:
                        if img_resized.mode != "RGB":
                            img_resized = img_resized.convert("RGB")
                        final_img.paste(img_resized, (x, y))

                    final_img.save(save_path, "PNG")
                    print(f"图像已缩放填充: {original_size} → {target_width}x{target_height}")
                return

            except ImportError:
                print("警告: PIL 未安装，无法缩放图像，保存原始尺寸")
            except Exception as e:
                print(f"警告: 图像缩放失败 ({e})，保存原始尺寸")

        # 直接保存原始数据
        with open(save_path, "wb") as f:
            f.write(raw_data)

    def _print_generation_info(self, prompts: dict, model_id: str, width: int = 0, height: int = 0):
        """打印生成信息"""
        print(f"\n{'='*70}")
        print(f"儿童涂色卡生成器")
        print(f"{'='*70}")

        if prompts.get("original_input"):
            orig = prompts["original_input"]
            print(f"输入: {orig['subject']} 在 {orig['location']} {orig['action']}")

        print(f"年龄模式: {prompts['age_mode']}")
        print(f"页面方向: {prompts.get('orientation', 'auto')}")
        print(f"请求尺寸: {width} x {height} px")
        print(f"模型: {model_id}")
        print(f"{'─'*70}")
        print(f"[USER PROMPT 预览]")
        print(prompts["user_prompt"][:500])
        if len(prompts["user_prompt"]) > 500:
            print("...")
        print(f"{'='*70}")


class ImagePostProcessor:
    """
    图像后处理器
    将生成的位图转换为矢量图和 PDF

    处理流程:
    PNG (像素图) → SVG (矢量图) → PDF (300 DPI A4)

    依赖:
    - potrace: 位图转矢量 (需安装: brew install potrace)
    - cairosvg: SVG转PDF (需安装: pip install cairosvg)
    """

    # A4 尺寸 @ 300 DPI (像素)
    A4_WIDTH_PX = 2480   # 210mm * 300 / 25.4
    A4_HEIGHT_PX = 3508  # 297mm * 300 / 25.4

    # A4 尺寸 (mm)
    A4_WIDTH_MM = 210
    A4_HEIGHT_MM = 297

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        """检查依赖是否已安装"""
        import shutil
        self.has_potrace = shutil.which("potrace") is not None

        # 检查 cairosvg 是否可用 (需要 cairo C 库)
        self.has_cairosvg = False
        try:
            import cairosvg
            # 尝试实际调用以验证 cairo 库是否存在
            cairosvg.svg2png(bytestring=b'<svg></svg>')
            self.has_cairosvg = True
        except Exception:
            self.has_cairosvg = False

        # 检查 PIL 是否可用
        try:
            from PIL import Image
            self.has_pil = True
        except ImportError:
            self.has_pil = False

        # 检查 reportlab 是否可用 (精确 A4 尺寸)
        try:
            from reportlab.lib.pagesizes import A4
            self.has_reportlab = True
        except ImportError:
            self.has_reportlab = False

    def png_to_svg(self, png_path: str, svg_path: Optional[str] = None) -> Optional[str]:
        """
        将 PNG 位图转换为 SVG 矢量图

        Args:
            png_path: 输入 PNG 文件路径
            svg_path: 输出 SVG 文件路径 (默认同名.svg)

        Returns:
            SVG 文件路径，失败返回 None
        """
        import subprocess

        if not self.has_potrace:
            print("警告: potrace 未安装，无法进行矢量化")
            print("安装方法: brew install potrace (macOS) 或 apt install potrace (Linux)")
            return None

        if svg_path is None:
            svg_path = png_path.rsplit(".", 1)[0] + ".svg"

        # PNG → PBM (potrace 需要 PBM 格式)
        pbm_path = png_path.rsplit(".", 1)[0] + ".pbm"

        try:
            # 使用 ImageMagick 或 PIL 转换为 PBM
            try:
                from PIL import Image
                img = Image.open(png_path).convert("L")  # 灰度
                # 二值化
                threshold = 200
                img = img.point(lambda x: 255 if x > threshold else 0, mode='1')
                img.save(pbm_path)
            except ImportError:
                # 尝试使用 ImageMagick
                subprocess.run(
                    ["convert", png_path, "-threshold", "50%", pbm_path],
                    check=True, capture_output=True
                )

            # PBM → SVG (使用 potrace)
            subprocess.run(
                ["potrace", "-s", "-o", svg_path, pbm_path],
                check=True, capture_output=True
            )

            # 清理临时文件
            if os.path.exists(pbm_path):
                os.remove(pbm_path)

            print(f"矢量化完成: {svg_path}")
            return svg_path

        except subprocess.CalledProcessError as e:
            print(f"矢量化失败: {e}")
            return None
        except Exception as e:
            print(f"矢量化出错: {e}")
            return None

    def svg_to_pdf(
        self,
        svg_path: str,
        pdf_path: Optional[str] = None,
        dpi: int = 300
    ) -> Optional[str]:
        """
        将 SVG 转换为 PDF (A4 尺寸, 300 DPI)

        Args:
            svg_path: 输入 SVG 文件路径
            pdf_path: 输出 PDF 文件路径 (默认同名.pdf)
            dpi: 输出分辨率

        Returns:
            PDF 文件路径，失败返回 None
        """
        if not self.has_cairosvg:
            print("警告: cairosvg 未安装，无法生成 PDF")
            print("安装方法: pip install cairosvg")
            return None

        if pdf_path is None:
            pdf_path = svg_path.rsplit(".", 1)[0] + ".pdf"

        try:
            import cairosvg

            # 计算缩放比例以适应 A4
            cairosvg.svg2pdf(
                url=svg_path,
                write_to=pdf_path,
                output_width=self.A4_WIDTH_MM * dpi / 25.4,
                output_height=self.A4_HEIGHT_MM * dpi / 25.4
            )

            print(f"PDF 生成完成: {pdf_path} (A4 @ {dpi} DPI)")
            return pdf_path

        except Exception as e:
            print(f"PDF 生成失败: {e}")
            return None

    def png_to_pdf(
        self,
        png_path: str,
        pdf_path: Optional[str] = None,
        vectorize: bool = True,
        dpi: int = 175
    ) -> Optional[str]:
        """
        完整后处理流程: PNG → (SVG) → PDF

        Args:
            png_path: 输入 PNG 文件路径
            pdf_path: 输出 PDF 文件路径
            vectorize: 是否先矢量化 (需要 potrace + cairosvg)
            dpi: 输出分辨率

        Returns:
            PDF 文件路径，失败返回 None
        """
        if pdf_path is None:
            pdf_path = png_path.rsplit(".", 1)[0] + ".pdf"

        # 方案1: PNG → SVG → PDF (矢量化，需要 potrace + cairosvg)
        if vectorize and self.has_potrace and self.has_cairosvg:
            svg_path = self.png_to_svg(png_path)
            if svg_path:
                result = self.svg_to_pdf(svg_path, pdf_path, dpi)
                if result:
                    return result

        # 方案2: 直接 PNG → PDF (使用 PIL，不需要 cairo)
        if self.has_pil:
            return self._png_to_pdf_direct(png_path, pdf_path, dpi)

        print("警告: 无法生成 PDF，请安装 Pillow: pip install Pillow")
        return None

    def _png_to_pdf_reportlab(self, png_path: str, pdf_path: str) -> Optional[str]:
        """使用 reportlab 生成精确 A4 尺寸的 PDF"""
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.pdfgen import canvas
        from PIL import Image

        img = Image.open(png_path)

        # 检测图像方向
        is_landscape = img.width > img.height

        # 选择页面尺寸 (A4 in points: 595.27 x 841.89)
        if is_landscape:
            page_size = landscape(A4)
        else:
            page_size = A4

        page_width, page_height = page_size

        # 计算图像绘制尺寸 (保持比例，居中)
        img_ratio = img.width / img.height
        page_ratio = page_width / page_height

        if img_ratio > page_ratio:
            # 图像更宽，以宽度为准
            draw_width = page_width
            draw_height = page_width / img_ratio
        else:
            # 图像更高，以高度为准
            draw_height = page_height
            draw_width = page_height * img_ratio

        # 居中位置
        x = (page_width - draw_width) / 2
        y = (page_height - draw_height) / 2

        # 创建 PDF
        c = canvas.Canvas(pdf_path, pagesize=page_size)
        c.drawImage(png_path, x, y, draw_width, draw_height)
        c.save()

        orientation = "横版" if is_landscape else "竖版"
        print(f"PDF 生成完成: {pdf_path} ({orientation} A4, 精确 210×297mm)")
        return pdf_path

    def _png_to_pdf_direct(
        self,
        png_path: str,
        pdf_path: str,
        dpi: int = 300
    ) -> Optional[str]:
        """直接将 PNG 嵌入 PDF，精确 A4 尺寸"""
        # 优先使用 reportlab (精确 A4)
        try:
            return self._png_to_pdf_reportlab(png_path, pdf_path)
        except ImportError:
            pass
        except Exception as e:
            print(f"reportlab 生成失败: {e}，尝试 PIL...")

        # 回退到 PIL
        try:
            from PIL import Image

            img = Image.open(png_path)

            # 检测图像方向
            is_landscape = img.width > img.height

            # 精确 A4 尺寸 (mm)
            if is_landscape:
                target_width_mm = self.A4_HEIGHT_MM  # 297mm
                target_height_mm = self.A4_WIDTH_MM  # 210mm
            else:
                target_width_mm = self.A4_WIDTH_MM   # 210mm
                target_height_mm = self.A4_HEIGHT_MM # 297mm

            # 计算精确像素尺寸
            a4_width = round(target_width_mm * dpi / 25.4)
            a4_height = round(target_height_mm * dpi / 25.4)

            # 计算精确 DPI 使 PDF 物理尺寸尽可能接近 A4
            exact_dpi_x = a4_width * 25.4 / target_width_mm
            exact_dpi_y = a4_height * 25.4 / target_height_mm
            exact_dpi = (exact_dpi_x + exact_dpi_y) / 2

            # 调整图像大小以适应 A4，保持比例
            img_ratio = img.width / img.height
            a4_ratio = a4_width / a4_height

            if img_ratio > a4_ratio:
                # 图像更宽，以宽度为准
                new_width = a4_width
                new_height = int(a4_width / img_ratio)
            else:
                # 图像更高，以高度为准
                new_height = a4_height
                new_width = int(a4_height * img_ratio)

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建白色 A4 背景
            a4_img = Image.new("RGB", (a4_width, a4_height), "white")

            # 居中放置
            x = (a4_width - new_width) // 2
            y = (a4_height - new_height) // 2

            # 处理透明通道
            if img_resized.mode == "RGBA":
                a4_img.paste(img_resized, (x, y), img_resized)
            else:
                if img_resized.mode != "RGB":
                    img_resized = img_resized.convert("RGB")
                a4_img.paste(img_resized, (x, y))

            # 保存为 PDF，使用精确 DPI 使物理尺寸为标准 A4
            a4_img.save(pdf_path, "PDF", resolution=exact_dpi)

            orientation = "横版" if is_landscape else "竖版"
            print(f"PDF 生成完成: {pdf_path} ({orientation} A4, {a4_width}x{a4_height}px @ {exact_dpi:.1f} DPI)")
            return pdf_path

        except ImportError:
            print("警告: PIL 未安装，无法生成 PDF")
            print("安装方法: pip install Pillow")
            return None
        except Exception as e:
            print(f"PDF 生成失败: {e}")
            return None

    def get_status(self) -> dict:
        """获取依赖状态"""
        return {
            "potrace": self.has_potrace,
            "cairosvg": self.has_cairosvg,
            "PIL": self.has_pil,
            "reportlab": self.has_reportlab,
            "可矢量化PDF": self.has_potrace and self.has_cairosvg,
            "精确A4 PDF": self.has_reportlab and self.has_pil,
            "可生成PDF": self.has_pil or self.has_cairosvg or self.has_reportlab
        }


def parse_chinese_input(text: str, age_mode: AgeMode = AgeMode.KIDS) -> SceneInput:
    """
    解析中文输入

    格式: 物体 在 某某地方 做什么
    示例: 小猫 在 花园里 追蝴蝶
    """
    subject = ""
    location = ""
    action = ""

    if " 在 " in text:
        parts = text.split(" 在 ", 1)
        subject = parts[0].strip()
        rest = parts[1].strip()

        # 查找地点后缀
        location_suffixes = ["里", "中", "上", "边", "间", "外", "下", "旁"]

        for i, char in enumerate(rest):
            if char in location_suffixes:
                location = rest[:i+1].strip()
                action = rest[i+1:].strip()
                break

        if not location:
            parts = rest.split(" ", 1)
            location = parts[0]
            action = parts[1] if len(parts) > 1 else ""
    else:
        parts = text.split()
        if len(parts) >= 3:
            subject = parts[0]
            location = parts[1]
            action = " ".join(parts[2:])
        elif len(parts) == 2:
            subject = parts[0]
            action = parts[1]
        else:
            subject = text

    return SceneInput(
        subject=subject,
        location=location,
        action=action,
        age_mode=age_mode
    )


# ============================================================
# 测试脚本
# ============================================================

def test_prompt_system():
    """测试提示词系统"""
    print("\n" + "="*70)
    print("测试提示词系统")
    print("="*70)

    prompt_system = ColoringPromptSystem()

    # 测试用例
    test_cases = [
        ("小猫 在 花园里 追蝴蝶", AgeMode.KIDS),
        ("小女孩 在 公园中 荡秋千", AgeMode.TODDLER),
        ("独角兽 在 云朵上 飞翔", AgeMode.KIDS),
        ("恐龙 在 森林里 吃东西", AgeMode.TODDLER),
    ]

    for text, age_mode in test_cases:
        scene = parse_chinese_input(text, age_mode)
        prompts = prompt_system.get_full_prompts(scene, mode="text")

        print(f"\n{'─'*70}")
        print(f"输入: {text}")
        print(f"年龄模式: {age_mode.value}")
        print(f"\n[解析结果]")
        print(f"  Subject: {scene.subject}")
        print(f"  Location: {scene.location}")
        print(f"  Action: {scene.action}")
        print(f"\n[USER PROMPT]")
        print(prompts["user_prompt"])

    print("\n" + "="*70)


def test_system_prompt():
    """显示系统提示词"""
    print("\n" + "="*70)
    print("系统提示词 (SYSTEM PROMPT) - 固定")
    print("="*70)

    prompt_system = ColoringPromptSystem()
    print(prompt_system.get_system_prompt())

    print("\n" + "="*70)


def test_image_to_coloring():
    """测试图生图提示词"""
    print("\n" + "="*70)
    print("测试图生图提示词")
    print("="*70)

    prompt_system = ColoringPromptSystem()

    for age_mode in [AgeMode.TODDLER, AgeMode.KIDS]:
        print(f"\n{'─'*70}")
        print(f"年龄模式: {age_mode.value}")
        print(f"\n[IMAGE TO COLORING PROMPT]")
        print(prompt_system.build_image_to_coloring_prompt(age_mode))

    print("\n" + "="*70)


def test_full_generation(api_key: Optional[str] = None):
    """测试完整生成流程"""
    print("\n" + "="*70)
    print("测试完整生成流程")
    print("="*70)

    if not api_key and not os.getenv("OPENROUTER_API_KEY"):
        print("\n[跳过] 未设置 OPENROUTER_API_KEY")
        print("请设置环境变量: export OPENROUTER_API_KEY='your-key'")
        return

    try:
        generator = OpenRouterImageGenerator(api_key)

        # 测试场景
        scene = parse_chinese_input("小猫 在 花园里 追蝴蝶", AgeMode.KIDS)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"output/coloring_{timestamp}.png"

        result = generator.generate_from_text(
            scene=scene,
            model="gpt5-image-mini",  # 默认使用 OpenAI (支持更大尺寸)
            save_path=save_path
        )

        if result["success"]:
            print(f"\n生成成功!")
            print(f"  PNG: {result['save_path']}")
            if result.get('pdf_path'):
                print(f"  PDF: {result['pdf_path']}")
        else:
            print(f"\n生成失败: {result['error']}")

    except Exception as e:
        print(f"\n测试出错: {e}")


def test_post_processor():
    """测试后处理器"""
    print("\n" + "="*70)
    print("测试后处理器 (PNG → SVG → PDF)")
    print("="*70)

    processor = ImagePostProcessor()
    status = processor.get_status()

    print("\n依赖状态:")
    for key, value in status.items():
        status_str = "✓ 已安装" if value else "✗ 未安装"
        print(f"  {key}: {status_str}")

    # 查找现有的 PNG 文件进行测试
    import glob
    png_files = glob.glob("output/*.png")

    if png_files:
        test_file = png_files[0]
        print(f"\n测试文件: {test_file}")

        # 测试矢量化
        if processor.has_potrace:
            svg_path = processor.png_to_svg(test_file)
            if svg_path:
                print(f"  SVG: {svg_path}")

        # 测试 PDF 生成
        pdf_path = processor.png_to_pdf(test_file, vectorize=True)
        if pdf_path:
            print(f"  PDF: {pdf_path}")
    else:
        print("\n没有找到测试用的 PNG 文件")
        print("请先运行 --test 生成图像")

    print("\n" + "="*70)


def interactive_mode():
    """交互式模式"""
    print("\n" + "="*70)
    print("儿童涂色卡生成器 - 交互模式")
    print("="*70)
    print("\n输入格式: 物体 在 某某地方 做什么")
    print("示例: 小猫 在 花园里 追蝴蝶")
    print("\n命令:")
    print("  toddler  - 切换到幼儿模式 (5岁以下)")
    print("  kids     - 切换到儿童模式 (6岁及以上)")
    print("  quit     - 退出")
    print()

    prompt_system = ColoringPromptSystem()
    current_age_mode = AgeMode.KIDS

    while True:
        try:
            mode_str = "幼儿" if current_age_mode == AgeMode.TODDLER else "儿童"
            user_input = input(f"[{mode_str}模式] 请输入: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            if user_input.lower() == 'toddler':
                current_age_mode = AgeMode.TODDLER
                print("已切换到幼儿模式 (5岁以下)")
                continue

            if user_input.lower() == 'kids':
                current_age_mode = AgeMode.KIDS
                print("已切换到儿童模式 (6岁及以上)")
                continue

            if not user_input:
                continue

            scene = parse_chinese_input(user_input, current_age_mode)
            prompts = prompt_system.get_full_prompts(scene, mode="text")

            print(f"\n{'─'*70}")
            print(f"[解析结果]")
            print(f"  Subject: {scene.subject}")
            print(f"  Location: {scene.location}")
            print(f"  Action: {scene.action}")
            print(f"  Age Mode: {scene.age_mode.value}")
            print(f"\n[生成的 USER PROMPT]")
            print(prompts["user_prompt"])
            print(f"{'─'*70}\n")

        except KeyboardInterrupt:
            print("\n再见!")
            break


if __name__ == "__main__":
    import sys

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║            儿童涂色卡生成器 - OpenRouter API                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  输入格式: 物体 在 某某地方 做什么                                    ║
║  示例: 小猫 在 花园里 追蝴蝶                                         ║
║                                                                      ║
║  4个控制变量:                                                        ║
║    1. Subject  (谁/什么)                                             ║
║    2. Location (在哪里)                                              ║
║    3. Action   (做什么)                                              ║
║    4. Age Mode (toddler: ≤5岁 / kids: ≥6岁)                         ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "--test":
            test_system_prompt()
            test_prompt_system()
            test_image_to_coloring()
            test_full_generation()

        elif cmd == "--interactive":
            interactive_mode()

        elif cmd == "--generate":
            if len(sys.argv) > 2:
                # 检查参数
                age_mode = AgeMode.KIDS
                orientation = Orientation.PORTRAIT
                model = "gemini-image"  # 默认模型 (支持更高分辨率)
                args = sys.argv[2:]

                if "--toddler" in args:
                    age_mode = AgeMode.TODDLER
                    args.remove("--toddler")
                elif "--kids" in args:
                    age_mode = AgeMode.KIDS
                    args.remove("--kids")

                if "--landscape" in args:
                    orientation = Orientation.LANDSCAPE
                    args.remove("--landscape")
                elif "--portrait" in args:
                    orientation = Orientation.PORTRAIT
                    args.remove("--portrait")

                # 解析 --model 参数
                for i, arg in enumerate(args):
                    if arg == "--model" and i + 1 < len(args):
                        model = args[i + 1]
                        args.remove("--model")
                        args.remove(model)
                        break

                scene_text = " ".join(args)
                scene = parse_chinese_input(scene_text, age_mode)
                scene.orientation = orientation

                generator = OpenRouterImageGenerator()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result = generator.generate_from_text(
                    scene=scene,
                    model=model,
                    save_path=f"output/coloring_{timestamp}.png"
                )
                print(json.dumps({
                    "success": result["success"],
                    "save_path": result.get("save_path"),
                    "pdf_path": result.get("pdf_path"),
                    "error": result.get("error")
                }, ensure_ascii=False, indent=2))

        elif cmd == "--system-prompt":
            test_system_prompt()

        elif cmd == "--post-test":
            test_post_processor()

        else:
            print("用法:")
            print("  python coloring_generator.py --test              # 运行所有测试 (含图像生成)")
            print("  python coloring_generator.py --interactive       # 交互模式")
            print("  python coloring_generator.py --system-prompt     # 显示系统提示词")
            print("  python coloring_generator.py --generate [选项] 小猫 在 花园里 追蝴蝶")
            print("  python coloring_generator.py --post-test         # 测试后处理器依赖状态")
            print("")
            print("生成选项:")
            print("  --toddler|--kids      年龄模式 (默认 kids)")
            print("  --portrait|--landscape 页面方向 (默认 portrait)")
            print("  --model <模型名>       指定模型 (默认 gpt5-image-mini)")
            print("")
            print("可用模型:")
            print("  gpt5-image-mini       OpenAI GPT-5 Image Mini (推荐, 支持 1536x1024)")
            print("  gpt5-image            OpenAI GPT-5 Image (高质量)")
            print("  gemini-image          Google Gemini 2.5 Flash (Nano Banana)")
            print("  nano-banana-pro       Google Gemini 3 Pro (最先进)")
            print("")
            print("示例:")
            print("  python coloring_generator.py --generate --model gpt5-image 小猫 在 花园里 追蝴蝶")
            print("  python coloring_generator.py --generate --landscape --toddler 恐龙 在 森林里 吃东西")
    else:
        # 默认运行测试
        test_prompt_system()
