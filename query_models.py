#!/usr/bin/env python3
"""
查询 OpenRouter API 获取图像生成模型的详细参数
"""

import os
import json
import requests
from pathlib import Path

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))


def get_model_info(model_id: str) -> dict:
    """获取单个模型的详细信息"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENROUTER_API_KEY")

    url = f"https://openrouter.ai/api/v1/models/{model_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json()
    else:
        # 尝试从模型列表中查找
        return get_model_from_list(model_id)


def get_model_from_list(model_id: str) -> dict:
    """从模型列表中查找"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        data = response.json()
        for model in data.get("data", []):
            if model.get("id") == model_id:
                return model
    return {"error": f"Model {model_id} not found"}


def get_all_image_models() -> list:
    """获取所有支持图像生成的模型"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        return []

    data = response.json()
    image_models = []

    for model in data.get("data", []):
        # 检查是否支持图像输出
        arch = model.get("architecture", {})
        modality = arch.get("modality", "")

        if "image" in modality.lower() and "->text+image" in modality.lower():
            image_models.append(model)

    return image_models


def print_model_info(model: dict):
    """格式化打印模型信息"""
    print("=" * 70)
    print(f"模型 ID: {model.get('id')}")
    print(f"名称: {model.get('name')}")
    print("-" * 70)

    # 描述
    desc = model.get("description", "N/A")
    if len(desc) > 300:
        desc = desc[:300] + "..."
    print(f"描述: {desc}")
    print()

    # 架构
    arch = model.get("architecture", {})
    print(f"输入/输出模态: {arch.get('modality', 'N/A')}")
    print(f"Tokenizer: {arch.get('tokenizer', 'N/A')}")
    print()

    # 价格
    pricing = model.get("pricing", {})
    print("价格:")
    print(f"  输入: ${float(pricing.get('prompt', 0)) * 1_000_000:.2f} / M tokens")
    print(f"  输出: ${float(pricing.get('completion', 0)) * 1_000_000:.2f} / M tokens")
    if pricing.get("image"):
        print(f"  图像: ${float(pricing.get('image', 0)):.4f} / 张")
    print()

    # 上下文长度
    print(f"上下文长度: {model.get('context_length', 'N/A')} tokens")

    # 支持的参数
    if model.get("supported_parameters"):
        print(f"支持的参数: {model.get('supported_parameters')}")

    # Top Provider
    top_provider = model.get("top_provider", {})
    if top_provider:
        print(f"Top Provider 上下文: {top_provider.get('context_length', 'N/A')}")

    print()


def main():
    print("=" * 70)
    print("OpenRouter 图像生成模型查询")
    print("=" * 70)
    print()

    # 要查询的模型
    target_models = [
        "openai/gpt-5-image",
        "openai/gpt-5-image-mini",
        "google/gemini-2.5-flash-image",
        "google/gemini-3-pro-image-preview",
    ]

    print(">>> 查询指定模型详情")
    print()

    for model_id in target_models:
        model = get_model_from_list(model_id)
        if "error" not in model:
            print_model_info(model)
        else:
            print(f"未找到模型: {model_id}")
            print()

    # 列出所有图像生成模型
    print("=" * 70)
    print(">>> 所有支持图像生成的模型")
    print("=" * 70)
    print()

    image_models = get_all_image_models()
    print(f"共找到 {len(image_models)} 个图像生成模型:\n")

    for m in image_models:
        pricing = m.get("pricing", {})
        img_price = float(pricing.get("image", 0))
        print(f"  {m.get('id')}")
        print(f"    名称: {m.get('name')}")
        print(f"    图像价格: ${img_price:.4f}/张" if img_price else "    图像价格: N/A")
        print()


if __name__ == "__main__":
    main()
