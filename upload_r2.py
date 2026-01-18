#!/usr/bin/env python3
"""
Batch upload WebP/PDF files to Cloudflare R2 (S3 compatible) and emit public links.
"""

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Iterable, List, Dict


def iter_files(root: Path, recursive: bool) -> Iterable[Path]:
    patterns = ["*.webp", "*.WEBP", "*.pdf", "*.PDF"]
    if root.is_file():
        return [root]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted(set(files))


def build_key(root: Path, file_path: Path, upload_path: str, mirror: bool) -> str:
    if mirror and root.is_dir():
        rel = file_path.relative_to(root)
        key = str(rel.as_posix())
    else:
        key = file_path.name
    if upload_path:
        return f"{upload_path.strip('/')}/{key}"
    return key


def to_public_url(domain: str, key: str) -> str:
    return f"{domain.rstrip('/')}/{key.lstrip('/')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload WebP/PDF to Cloudflare R2")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("--bucket", required=True, help="R2 bucket name")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="R2 endpoint, e.g. https://<accountid>.r2.cloudflarestorage.com",
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Public domain used to build links, e.g. https://cdn.example.com",
    )
    parser.add_argument(
        "--upload-path",
        default="",
        help="Prefix path within bucket, e.g. uploads",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Preserve input directory structure under upload path",
    )
    parser.add_argument(
        "--output",
        default="r2_links.json",
        help="Output JSON file with uploaded links",
    )
    parser.add_argument(
        "--csv",
        default="r2_links.csv",
        help="Output CSV file with uploaded links",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without uploading",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Env file to load (default: .env)",
    )
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
    except Exception:
        load_dotenv = None

    if load_dotenv:
        load_dotenv(args.env_file)

    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing R2 credentials. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY."
        )

    input_path = Path(args.input)
    files = iter_files(input_path, args.recursive)
    if not files:
        print(f"[warn] No .webp/.pdf files found under {input_path}")
        return

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required. Install with: pip install boto3") from exc

    s3 = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    links: List[Dict[str, str]] = []
    for file_path in files:
        key = build_key(input_path, file_path, args.upload_path, args.mirror)
        content_type, _ = mimetypes.guess_type(file_path.name)
        extra_args = {"ContentType": content_type or "application/octet-stream"}

        if args.dry_run:
            print(f"[dry-run] {file_path} -> {key}")
        else:
            s3.upload_file(str(file_path), args.bucket, key, ExtraArgs=extra_args)
            print(f"[ok] {file_path.name} -> {key}")

        links.append(
            {
                "file": str(file_path),
                "key": key,
                "url": to_public_url(args.domain, key),
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)

    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("file,key,url\n")
        for item in links:
            f.write(f"{item['file']},{item['key']},{item['url']}\n")

    print(f"[done] {len(links)} files")
    print(f"[links] JSON: {args.output}")
    print(f"[links] CSV : {args.csv}")


if __name__ == "__main__":
    main()

  # .env 里需要：
  # API_KEY=...
  # API_BASE_URL=https://yunwu.ai/v1beta

#   python generate_alt_from_r2.py r2_links.csv \
#     --output-csv r2_links_with_alt.csv \
#     --model gemini-2.5-flash-all

#   如果你想直接写死到你给的接口，也可以用：

#   --base-url https://yunwu.ai/v1beta

#   脚本位置

#   - generate_alt_from_r2.py

#   需要我再加并发（提高速度）或失败重试吗？