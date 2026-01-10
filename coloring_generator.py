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


class AgeMode(Enum):
    """年龄模式"""
    TODDLER = "toddler"  # 5岁以下
    KIDS = "kids"        # 6岁及以上


@dataclass
class SceneInput:
    """场景输入结构 - 4个核心变量"""
    subject: str                           # 物体/主体 (谁/什么)
    location: str                          # 地点 (在哪里)
    action: str                            # 动作 (做什么)
    age_mode: AgeMode = AgeMode.KIDS       # 年龄模式


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
    SYSTEM_PROMPT = """[ROLE]
You are a highly experienced professional children's coloring book illustrator
and print design expert.

You specialize in creating high-quality, printable black-and-white coloring
pages for children of different age groups.

[CORE PRIORITY]
Real-world usability for children > Printability > Artistic complexity

In all situations, the result must be:
- Easy to color
- Safe from color bleeding
- Ready for direct printing

[HARD RULES — MUST FOLLOW]

1. Line Quality and Structure
- All lines must be clean, smooth, and continuous
- All outlines must be fully closed with no gaps of any kind
- Line thickness must be uniform throughout the entire image
- No double lines, no overlapping strokes, no dashed lines
- Line intersections must be clear single-point intersections
  (T-junction or clean cross only)

2. Fill and Shading
- Black outlines on a pure white background only
- No solid black areas or filled regions
- No grayscale, shadows, gradients, or textures
- Dark-colored objects must be represented using outlines only
- Eyes and small details must be simple dots or minimal shapes

3. Shape Design and Complexity
- All shapes must be simplified and child-friendly
- Each coloring area must be large enough for comfortable coloring
- Prefer rounded, smooth shapes over sharp or narrow ones
- Avoid tiny gaps, thin corridors, or decorative micro-details

4. Layout and Output Format
- Page composition must match A4 print proportions
- Layout must be centered and well-balanced
- Clear white margins must be preserved around the page
- Background must be pure white
- Overall style must resemble clean vector line art
  found in professional children's coloring books

[FORBIDDEN ELEMENTS]
- Any text, letters, or numbers
- Logos, watermarks, or signatures
- Color, grayscale, transparency
- Photorealistic details or complex textures

[SELF-CHECK BEFORE FINAL OUTPUT]
Before finalizing, ensure:
- Every enclosed region can be safely colored independently
- No broken lines or open shapes exist
- No dark filled areas appear
- Line weight is consistent and print-friendly
- Overall complexity matches the target age group"""

    # ================================================================
    # 年龄模式提示词
    # ================================================================
    AGE_MODE_PROMPTS = {
        AgeMode.TODDLER: """[AGE MODE: TODDLER / PRESCHOOL — UNDER 5]

- Use very large and simple shapes
- Minimize the number of objects
- Avoid small enclosed areas
- Use bold, rounded forms
- Keep the scene extremely clear and easy to understand""",

        AgeMode.KIDS: """[AGE MODE: KIDS — 6 YEARS OLD AND ABOVE]

- Use moderately large coloring areas
- Allow a moderate level of detail
- Clearly separate different objects
- Keep the design clean and readable
- Avoid excessive complexity or tiny patterns"""
    }

    # ================================================================
    # 用户提示词模板 - 文本生成涂色卡
    # ================================================================
    TEXT_TO_COLORING_TEMPLATE = """Create a children's coloring page based on the following description.

Subject:
{subject}

Location:
{location}

Action:
{action}

{age_mode_prompt}

[STYLE REQUIREMENTS]
- Black and white line art only
- Clean, closed outlines
- Uniform line thickness
- Vector-style illustration
- No shading, no color, no filled areas

[PAGE FORMAT]
- A4 printable layout
- Automatically choose portrait or landscape orientation
- Centered composition with clear margins

The final result must look like a professional children's coloring book page
that is easy, safe, and enjoyable for children to color."""

    # ================================================================
    # 用户提示词模板 - 图片转涂色卡
    # ================================================================
    IMAGE_TO_COLORING_TEMPLATE = """Convert the provided image into a children's coloring book page.

IMPORTANT:
Do NOT trace the image directly.
Redraw it as simplified, clean vector-style line art suitable for children.

{age_mode_prompt}

[REQUIRED TRANSFORMATION RULES]
- Simplify all shapes
- Close all outlines completely
- Remove unnecessary details
- Fix any broken or overlapping lines
- Eliminate all shading, textures, and filled areas

[OUTPUT REQUIREMENTS]
- Black outlines on a pure white background
- Uniform line thickness
- A4 printable layout
- Clean, child-friendly composition

The final result must be a high-quality children's coloring page
that can be printed and colored safely without color bleeding."""

    # ================================================================
    # 中文词汇翻译映射
    # ================================================================
    SUBJECT_MAPPINGS = {
        # 动物类
        "小猫": "a cute little kitten",
        "猫": "a cat",
        "小狗": "a cute puppy",
        "狗": "a dog",
        "兔子": "a rabbit",
        "小兔子": "a baby bunny",
        "熊猫": "a panda bear",
        "小熊": "a teddy bear",
        "小鸟": "a little bird",
        "蝴蝶": "a butterfly",
        "小鱼": "a fish",
        "小马": "a pony",
        "独角兽": "a unicorn",
        "恐龙": "a friendly dinosaur",
        "小象": "a baby elephant",
        "小狮子": "a lion cub",
        "长颈鹿": "a giraffe",
        "猴子": "a monkey",
        "企鹅": "a penguin",
        "海豚": "a dolphin",
        "乌龟": "a turtle",
        "瓢虫": "a ladybug",
        "蜜蜂": "a bee",
        "蜗牛": "a snail",
        "青蛙": "a frog",
        "螃蟹": "a crab",
        "章鱼": "an octopus",
        "老虎": "a tiger",
        "斑马": "a zebra",
        "考拉": "a koala",

        # 人物类
        "小女孩": "a little girl",
        "小男孩": "a little boy",
        "公主": "a princess",
        "王子": "a prince",
        "仙女": "a fairy",
        "小朋友": "a child",
        "宝宝": "a baby",
        "超人": "a superhero",
        "海盗": "a pirate",
        "宇航员": "an astronaut",
        "消防员": "a firefighter",
        "医生": "a doctor",
        "老师": "a teacher",
        "厨师": "a chef",

        # 交通工具
        "汽车": "a car",
        "火车": "a train",
        "飞机": "an airplane",
        "轮船": "a boat",
        "火箭": "a rocket",
        "自行车": "a bicycle",
        "公交车": "a bus",
        "消防车": "a fire truck",
        "挖掘机": "an excavator",
        "直升机": "a helicopter",
        "热气球": "a hot air balloon",

        # 其他
        "房子": "a house",
        "城堡": "a castle",
        "花": "a flower",
        "树": "a tree",
        "太阳": "the sun",
        "月亮": "the moon",
        "星星": "stars",
        "彩虹": "a rainbow",
        "蛋糕": "a birthday cake",
        "气球": "balloons",
        "风筝": "a kite",
        "雪人": "a snowman",
        "圣诞树": "a Christmas tree",
    }

    LOCATION_MAPPINGS = {
        "花园里": "in a garden",
        "花园中": "in a garden",
        "公园里": "in a park",
        "公园中": "in a park",
        "森林里": "in a forest",
        "森林中": "in a forest",
        "草地上": "on a meadow",
        "河边": "by the river",
        "湖边": "near a lake",
        "海边": "at the beach",
        "沙滩上": "on the beach",
        "山上": "on a mountain",
        "天空中": "in the sky",
        "云朵上": "on the clouds",
        "房间里": "in a room",
        "卧室里": "in a bedroom",
        "厨房里": "in a kitchen",
        "客厅里": "in a living room",
        "学校里": "at school",
        "操场上": "on a playground",
        "农场里": "on a farm",
        "动物园里": "at the zoo",
        "游乐园里": "at an amusement park",
        "城堡里": "in a castle",
        "太空中": "in outer space",
        "水里": "underwater",
        "海底": "under the sea",
        "雪地上": "in the snow",
        "游泳池里": "in a swimming pool",
        "马路上": "on the street",
        "树上": "in a tree",
        "花丛中": "among the flowers",
        "田野里": "in a field",
        "竹林里": "in a bamboo forest",
    }

    ACTION_MAPPINGS = {
        "追蝴蝶": "chasing butterflies",
        "玩球": "playing with a ball",
        "读书": "reading a book",
        "唱歌": "singing",
        "跳舞": "dancing",
        "画画": "painting",
        "吃东西": "eating",
        "睡觉": "sleeping",
        "跑步": "running",
        "游泳": "swimming",
        "飞翔": "flying",
        "骑车": "riding a bicycle",
        "荡秋千": "swinging on a swing",
        "放风筝": "flying a kite",
        "堆沙堡": "building a sandcastle",
        "摘花": "picking flowers",
        "浇花": "watering flowers",
        "钓鱼": "fishing",
        "野餐": "having a picnic",
        "过生日": "celebrating a birthday",
        "开派对": "having a party",
        "做游戏": "playing games",
        "拥抱": "hugging",
        "微笑": "smiling",
        "看星星": "looking at the stars",
        "赏月": "watching the moon",
        "滑滑梯": "sliding down a slide",
        "爬树": "climbing a tree",
        "吹泡泡": "blowing bubbles",
        "吃冰淇淋": "eating ice cream",
        "吃竹子": "eating bamboo",
        "喝水": "drinking water",
        "追逐": "chasing",
        "玩耍": "playing",
        "散步": "taking a walk",
        "跳跃": "jumping",
        "挖沙子": "digging in the sand",
        "堆雪人": "building a snowman",
        "滑雪": "skiing",
        "溜冰": "ice skating",
    }

    def __init__(self):
        pass

    def _translate_element(self, text: str, mapping: dict) -> str:
        """翻译单个元素"""
        if not text:
            return ""

        # 精确匹配
        if text in mapping:
            return mapping[text]

        # 部分匹配
        for key, value in mapping.items():
            if key in text:
                return value

        # 无匹配时返回原文
        return text

    def get_system_prompt(self) -> str:
        """获取系统提示词（固定）"""
        return self.SYSTEM_PROMPT

    def build_text_to_coloring_prompt(self, scene: SceneInput) -> str:
        """
        构建文本生成涂色卡的用户提示词

        Args:
            scene: 包含 subject, location, action, age_mode 的场景输入

        Returns:
            str: 完整的用户提示词
        """
        # 翻译各个元素
        subject_en = self._translate_element(scene.subject, self.SUBJECT_MAPPINGS)
        location_en = self._translate_element(scene.location, self.LOCATION_MAPPINGS)
        action_en = self._translate_element(scene.action, self.ACTION_MAPPINGS)

        # 获取年龄模式提示词
        age_mode_prompt = self.AGE_MODE_PROMPTS[scene.age_mode]

        # 填充模板
        user_prompt = self.TEXT_TO_COLORING_TEMPLATE.format(
            subject=subject_en,
            location=location_en,
            action=action_en,
            age_mode_prompt=age_mode_prompt
        )

        return user_prompt

    def build_image_to_coloring_prompt(self, age_mode: AgeMode = AgeMode.KIDS) -> str:
        """
        构建图片转涂色卡的用户提示词

        Args:
            age_mode: 年龄模式

        Returns:
            str: 完整的用户提示词
        """
        age_mode_prompt = self.AGE_MODE_PROMPTS[age_mode]

        user_prompt = self.IMAGE_TO_COLORING_TEMPLATE.format(
            age_mode_prompt=age_mode_prompt
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
            user_prompt = self.build_image_to_coloring_prompt(scene.age_mode)

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "age_mode": scene.age_mode.value,
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

    SUPPORTED_MODELS = {
        # Flux 系列 (推荐)
        "flux-schnell": "black-forest-labs/flux-schnell",
        "flux-pro": "black-forest-labs/flux-pro",
        "flux-dev": "black-forest-labs/flux-1.1-pro",

        # Stable Diffusion 系列
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd3": "stabilityai/stable-diffusion-3",

        # DALL-E 系列
        "dalle3": "openai/dall-e-3",

        # 其他
        "playground": "playgroundai/playground-v2.5-1024px-aesthetic",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 OPENROUTER_API_KEY 环境变量或传入 api_key 参数")

        self.base_url = "https://openrouter.ai/api/v1"
        self.prompt_system = ColoringPromptSystem()

    def generate_from_text(
        self,
        scene: SceneInput,
        model: str = "flux-schnell",
        width: int = 1024,
        height: int = 1448,  # A4 比例 (1:1.414)
        save_path: Optional[str] = None
    ) -> dict:
        """
        文本生成涂色卡

        Args:
            scene: 场景输入 (subject, location, action, age_mode)
            model: 模型名称
            width: 图像宽度
            height: 图像高度 (默认 A4 比例)
            save_path: 保存路径
        """
        prompts = self.prompt_system.get_full_prompts(scene, mode="text")
        return self._generate(prompts, model, width, height, save_path)

    def generate_from_image(
        self,
        image_path: str,
        age_mode: AgeMode = AgeMode.KIDS,
        model: str = "flux-schnell",
        width: int = 1024,
        height: int = 1448,
        save_path: Optional[str] = None
    ) -> dict:
        """
        图片转涂色卡

        Args:
            image_path: 源图片路径
            age_mode: 年龄模式
            model: 模型名称
            width: 输出宽度
            height: 输出高度
            save_path: 保存路径
        """
        scene = SceneInput(subject="", location="", action="", age_mode=age_mode)
        prompts = self.prompt_system.get_full_prompts(scene, mode="image")
        prompts["source_image"] = image_path

        return self._generate(prompts, model, width, height, save_path, image_path)

    def _generate(
        self,
        prompts: dict,
        model: str,
        width: int,
        height: int,
        save_path: Optional[str],
        image_path: Optional[str] = None
    ) -> dict:
        """执行图像生成"""
        model_id = self.SUPPORTED_MODELS.get(model, model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://coloring-generator.local",
            "X-Title": "Children Coloring Page Generator"
        }

        # 构建请求
        if "dall-e" in model_id:
            payload = self._build_dalle_payload(prompts, model_id, width, height)
            endpoint = f"{self.base_url}/chat/completions"
        else:
            payload = self._build_sd_payload(prompts, model_id, width, height, image_path)
            endpoint = f"{self.base_url}/images/generations"

        self._print_generation_info(prompts, model_id)

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()

            image_data = self._extract_image(result, model_id)

            if save_path and image_data:
                self._save_image(image_data, save_path)
                print(f"\n图像已保存至: {save_path}")

            return {
                "success": True,
                "prompts": prompts,
                "model": model_id,
                "image_data": image_data,
                "save_path": save_path,
                "raw_response": result
            }

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "prompts": prompts,
                "model": model_id
            }

    def _build_sd_payload(
        self,
        prompts: dict,
        model_id: str,
        width: int,
        height: int,
        image_path: Optional[str] = None
    ) -> dict:
        """构建 SD/Flux 请求体"""
        # 将 system prompt 和 user prompt 合并
        full_prompt = f"{prompts['system_prompt']}\n\n{prompts['user_prompt']}"

        payload = {
            "model": model_id,
            "prompt": full_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "num_outputs": 1,
        }

        # 如果是图生图，添加源图像
        if image_path:
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            payload["image"] = image_b64
            payload["strength"] = 0.75

        return payload

    def _build_dalle_payload(
        self,
        prompts: dict,
        model_id: str,
        width: int,
        height: int
    ) -> dict:
        """构建 DALL-E 请求体"""
        # 选择最接近的尺寸
        if height > width:
            size = "1024x1792"
        elif width > height:
            size = "1792x1024"
        else:
            size = "1024x1024"

        return {
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompts["system_prompt"]},
                {"role": "user", "content": prompts["user_prompt"]}
            ],
            "size": size,
            "quality": "standard",
            "n": 1
        }

    def _extract_image(self, result: dict, model_id: str) -> Optional[str]:
        """提取图像数据"""
        try:
            if "dall-e" in model_id:
                return result.get("data", [{}])[0].get("url")
            else:
                data = result.get("data", [{}])[0]
                return data.get("b64_json") or data.get("url")
        except (IndexError, KeyError):
            return None

    def _save_image(self, image_data: str, save_path: str):
        """保存图像"""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        if image_data.startswith("http"):
            response = requests.get(image_data, timeout=60)
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            with open(save_path, "wb") as f:
                f.write(base64.b64decode(image_data))

    def _print_generation_info(self, prompts: dict, model_id: str):
        """打印生成信息"""
        print(f"\n{'='*70}")
        print(f"儿童涂色卡生成器")
        print(f"{'='*70}")

        if prompts.get("original_input"):
            orig = prompts["original_input"]
            print(f"输入: {orig['subject']} 在 {orig['location']} {orig['action']}")

        print(f"年龄模式: {prompts['age_mode']}")
        print(f"模型: {model_id}")
        print(f"{'─'*70}")
        print(f"[USER PROMPT 预览]")
        print(prompts["user_prompt"][:500])
        if len(prompts["user_prompt"]) > 500:
            print("...")
        print(f"{'='*70}")


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
        print(f"  Subject: {scene.subject} -> {prompt_system._translate_element(scene.subject, prompt_system.SUBJECT_MAPPINGS)}")
        print(f"  Location: {scene.location} -> {prompt_system._translate_element(scene.location, prompt_system.LOCATION_MAPPINGS)}")
        print(f"  Action: {scene.action} -> {prompt_system._translate_element(scene.action, prompt_system.ACTION_MAPPINGS)}")
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
            model="flux-schnell",
            save_path=save_path
        )

        if result["success"]:
            print(f"\n生成成功! 保存路径: {result['save_path']}")
        else:
            print(f"\n生成失败: {result['error']}")

    except Exception as e:
        print(f"\n测试出错: {e}")


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
                # 检查年龄模式参数
                age_mode = AgeMode.KIDS
                args = sys.argv[2:]

                if "--toddler" in args:
                    age_mode = AgeMode.TODDLER
                    args.remove("--toddler")
                elif "--kids" in args:
                    age_mode = AgeMode.KIDS
                    args.remove("--kids")

                scene_text = " ".join(args)
                scene = parse_chinese_input(scene_text, age_mode)

                generator = OpenRouterImageGenerator()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result = generator.generate_from_text(
                    scene=scene,
                    save_path=f"output/coloring_{timestamp}.png"
                )
                print(json.dumps({
                    "success": result["success"],
                    "save_path": result.get("save_path"),
                    "error": result.get("error")
                }, ensure_ascii=False, indent=2))

        elif cmd == "--system-prompt":
            test_system_prompt()

        else:
            print("用法:")
            print("  python coloring_generator.py --test              # 运行所有测试")
            print("  python coloring_generator.py --interactive       # 交互模式")
            print("  python coloring_generator.py --system-prompt     # 显示系统提示词")
            print("  python coloring_generator.py --generate [--toddler|--kids] 小猫 在 花园里 追蝴蝶")
    else:
        # 默认运行测试
        test_prompt_system()
