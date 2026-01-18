#!/usr/bin/env python3
"""
Generate alt text for R2 links CSV using Gemini vision (generateContent).
Only processes .webp entries and appends a SEO phrase.
"""

import argparse
import base64
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse


SEO_PHRASE = "AI-generated coloring page"


def load_env(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(env_file)


def extract_url(row: Dict[str, str]) -> str:
    url = row.get("url", "")
    return url.strip()


def extract_basename(row: Dict[str, str]) -> str:
    url = row.get("url", "")
    key = row.get("key", "")
    file_path = row.get("file", "")

    for value in (url, key, file_path):
        if value:
            if value.startswith("http"):
                value = urlparse(value).path
            name = Path(value).name
            if name:
                return name
    return ""


def build_prompt() -> str:
    return (
        "You write concise alt text for a kids coloring page website. "
        "Generate one alt text line in English, about 10 words. "
        "Do not mention file names, formats, or URLs. "
        f"The line must end with the exact phrase '{SEO_PHRASE}'."
    )


def fetch_image_base64(image_url: str) -> str:
    import requests

    resp = requests.get(image_url, timeout=60)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("ascii")


def call_gemini_generate(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_url: Optional[str],
) -> str:
    import requests

    endpoint = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    parts = [{"text": prompt}]
    if image_url:
        image_b64 = fetch_image_base64(image_url)
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/webp",
                    "data": image_b64,
                }
            }
        )
    payload = {"contents": [{"role": "user", "parts": parts}]}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def normalize_alt(text: str) -> str:
    alt = " ".join(text.split())
    if not alt.endswith(SEO_PHRASE):
        alt = f"{alt.rstrip('.')} {SEO_PHRASE}"
    return alt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate alt text for R2 links CSV")
    parser.add_argument("input_csv", help="Input r2_links.csv")
    parser.add_argument(
        "--output-csv",
        default="r2_links_with_alt.csv",
        help="Output CSV with alt text",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-all",
        help="Gemini model id (default: gemini-2.5-flash-all)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Gemini base URL, e.g. https://yunwu.ai/v1beta",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file to load (default: .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample prompts without calling API",
    )
    args = parser.parse_args()

    load_env(args.env_file)
    api_key = os.getenv("API_KEY")
    base_url = args.base_url or os.getenv("API_BASE_URL", "https://yunwu.ai/v1beta")
    if not api_key:
        raise RuntimeError("Missing API_KEY in environment or .env")

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    rows: List[Dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").lower()
            key = (row.get("key") or "").lower()
            if url.endswith(".webp") or key.endswith(".webp"):
                rows.append(row)

    if not rows:
        print("[warn] No .webp rows found in CSV.")
        return

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        basename = extract_basename(row)
        image_url = extract_url(row)
        if not image_url:
            print(f"[skip] {basename} (missing url)")
            continue
        prompt = build_prompt()

        if args.dry_run:
            alt = f"Sample alt text {SEO_PHRASE}"
        else:
            alt = call_gemini_generate(
                base_url, api_key, args.model, prompt, image_url
            )
            alt = normalize_alt(alt)

        out_row = dict(row)
        out_row["alt"] = alt
        out_rows.append(out_row)
        print(f"[ok] {basename} -> {alt}")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(out_rows[0].keys())
        if "alt" not in fieldnames:
            fieldnames.append("alt")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[done] {len(out_rows)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
