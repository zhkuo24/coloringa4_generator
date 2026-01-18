#!/usr/bin/env python3
"""
Batch convert PNG files to A4-sized WebP and PDF.

Reference behavior: A4 canvas at 300 DPI, center-fit image, white background.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple


def mm_to_pixels(mm: float, dpi: int) -> int:
    return round((mm * dpi) / 25.4)


def a4_dimensions(dpi: int, is_landscape: bool) -> Tuple[int, int]:
    width_mm, height_mm = (297, 210) if is_landscape else (210, 297)
    return mm_to_pixels(width_mm, dpi), mm_to_pixels(height_mm, dpi)


def load_image(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required. Install with: pip install pillow") from exc
    return Image.open(path)


def to_a4_canvas(img, dpi: int):
    from PIL import Image

    is_landscape = img.width > img.height
    a4_w, a4_h = a4_dimensions(dpi, is_landscape)

    ratio = min(a4_w / img.width, a4_h / img.height)
    new_w = max(1, int(round(img.width * ratio)))
    new_h = max(1, int(round(img.height * ratio)))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (a4_w, a4_h), "white")
    x = (a4_w - new_w) // 2
    y = (a4_h - new_h) // 2

    if resized.mode in ("RGBA", "LA") or ("transparency" in resized.info):
        rgba = resized.convert("RGBA")
        canvas.paste(rgba, (x, y), rgba)
    else:
        canvas.paste(resized.convert("RGB"), (x, y))

    return canvas, ("landscape" if is_landscape else "portrait")


def iter_pngs(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        return [path]
    if recursive:
        return [p for p in path.rglob("*.png")] + [p for p in path.rglob("*.PNG")]
    return [p for p in path.glob("*.png")] + [p for p in path.glob("*.PNG")]


def convert_one(
    png_path: Path,
    out_dir: Path,
    dpi: int,
    webp_quality: int,
    overwrite: bool,
    write_a4_png: bool,
    pdf_jpeg: bool,
    pdf_jpeg_quality: int,
) -> None:
    img = load_image(png_path)
    a4_img, orientation = to_a4_canvas(img, dpi)

    rel_name = png_path.stem
    out_webp = out_dir / f"{rel_name}.webp"
    out_pdf = out_dir / f"{rel_name}.pdf"
    out_png = out_dir / f"{rel_name}.png"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        if out_webp.exists() and out_pdf.exists() and (not write_a4_png or out_png.exists()):
            print(f"[skip] {png_path.name} (outputs exist)")
            return

    if write_a4_png:
        a4_img.save(out_png, "PNG")

    a4_img.save(out_webp, "WEBP", quality=webp_quality, method=6)
    if pdf_jpeg:
        rgb = a4_img.convert("RGB")
        rgb.save(
            out_pdf,
            "PDF",
            resolution=dpi,
            quality=pdf_jpeg_quality,
            optimize=True,
        )
    else:
        a4_img.save(out_pdf, "PDF", resolution=dpi)

    print(
        f"[ok] {png_path.name} -> {out_webp.name}, {out_pdf.name} ({orientation}, {dpi} DPI)"
    )


def batch_convert(
    input_path: Path,
    output_dir: Path,
    dpi: int = 300,
    webp_quality: int = 85,
    recursive: bool = False,
    overwrite: bool = False,
    write_a4_png: bool = False,
    mirror: bool = False,
    pdf_jpeg: bool = True,
    pdf_jpeg_quality: int = 90,
) -> None:
    pngs = sorted(iter_pngs(input_path, recursive))
    if not pngs:
        print(f"[warn] No PNG files found under {input_path}")
        return

    for png_path in pngs:
        if mirror and input_path.is_dir():
            rel_parent = png_path.parent.relative_to(input_path)
            out_dir = output_dir / rel_parent
        else:
            out_dir = output_dir
        convert_one(
            png_path=png_path,
            out_dir=out_dir,
            dpi=dpi,
            webp_quality=webp_quality,
            overwrite=overwrite,
            write_a4_png=write_a4_png,
            pdf_jpeg=pdf_jpeg,
            pdf_jpeg_quality=pdf_jpeg_quality,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert PNG to A4 WebP/PDF")
    parser.add_argument("input", help="Input PNG file or directory")
    parser.add_argument(
        "-o",
        "--output",
        default="output_converted",
        help="Output directory (default: output_converted)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="A4 DPI (default: 300)")
    parser.add_argument(
        "--webp-quality", type=int, default=85, help="WebP quality (default: 85)"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    parser.add_argument(
        "--write-a4-png",
        action="store_true",
        help="Also write A4 PNG alongside WebP/PDF",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror input folder structure under output directory",
    )
    parser.add_argument(
        "--no-pdf-jpeg",
        action="store_true",
        help="Disable JPEG compression for PDF",
    )
    parser.add_argument(
        "--pdf-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for PDF (default: 90)",
    )

    args = parser.parse_args()
    batch_convert(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        dpi=args.dpi,
        webp_quality=args.webp_quality,
        recursive=args.recursive,
        overwrite=args.overwrite,
        write_a4_png=args.write_a4_png,
        mirror=args.mirror,
        pdf_jpeg=not args.no_pdf_jpeg,
        pdf_jpeg_quality=args.pdf_jpeg_quality,
    )


if __name__ == "__main__":
    main()
