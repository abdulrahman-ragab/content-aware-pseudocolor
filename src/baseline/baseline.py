"""Fixed Viridis pseudo-color baseline for 8-bit grayscale PNG images."""

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def apply_viridis(grayscale: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Map unchanged grayscale intensities to RGB using OpenCV's fixed Viridis LUT."""
    if grayscale.dtype != np.uint8:
        raise ValueError(f"Expected uint8 pixels, got {grayscale.dtype}.")
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("Expected a non-empty, two-dimensional grayscale image.")
    bgr = cv2.applyColorMap(np.ascontiguousarray(grayscale), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_grayscale(path: Path) -> NDArray[np.uint8]:
    """Read grayscale pixels, including lossless opaque RGB/RGBA representations."""
    # Byte-based decoding also supports Windows paths containing non-ASCII characters.
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Image file is empty: {path}")
    grayscale = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if grayscale is None:
        raise ValueError(f"Cannot decode image: {path}")
    if grayscale.dtype != np.uint8:
        raise ValueError(
            f"Expected an 8-bit grayscale image: {path} "
            f"(dtype={grayscale.dtype}, shape={grayscale.shape})."
        )
    if grayscale.ndim == 3 and grayscale.shape[2] in (3, 4):
        if not (
            np.array_equal(grayscale[:, :, 0], grayscale[:, :, 1])
            and np.array_equal(grayscale[:, :, 0], grayscale[:, :, 2])
        ):
            raise ValueError(f"Expected grayscale pixels with identical RGB channels: {path}")
        if grayscale.shape[2] == 4 and not np.all(grayscale[:, :, 3] == 255):
            raise ValueError(f"Expected fully opaque grayscale pixels; found transparency: {path}")
        LOGGER.info(
            "Reading %s as grayscale: identical RGB channels%s; intensities unchanged.",
            path.name,
            " and fully opaque alpha" if grayscale.shape[2] == 4 else "",
        )
        return grayscale[:, :, 0].copy()
    if grayscale.ndim != 2:
        raise ValueError(
            f"Expected an 8-bit grayscale image: {path} "
            f"(dtype={grayscale.dtype}, shape={grayscale.shape})."
        )
    return grayscale


def save_rgb(path: Path, rgb: NDArray[np.uint8]) -> None:
    """Encode RGB pixels as a lossless PNG, reporting encoding or filesystem errors."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise OSError(f"Could not encode PNG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def save_comparison(
    path: Path, grayscale: NDArray[np.uint8], rgb: NDArray[np.uint8]
) -> None:
    """Save an unscaled side-by-side example with a fixed 0-255 Viridis colorbar."""
    height, width = grayscale.shape
    panel_width = max(width, 256)
    canvas = np.full((height + 100, panel_width * 2, 3), 255, dtype=np.uint8)
    canvas[40 : height + 40, :width] = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    canvas[40 : height + 40, panel_width : panel_width + width] = rgb
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    canvas[height + 50 : height + 70, panel_width : panel_width + 256] = (
        apply_viridis(ramp)
    )
    for text, position in [
        ("Original grayscale", (8, 25)),
        ("Fixed Viridis", (panel_width + 8, 25)),
        ("Intensity: 0", (panel_width, height + 90)),
        ("255", (panel_width + 220, height + 90)),
    ]:
        cv2.putText(
            canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    save_rgb(path, canvas)


def run_baseline(
    input_dir: Path, output_dir: Path, gallery_size: int = 10
) -> Path:
    """Process sorted PNGs recursively; write a manifest only after a successful run."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if gallery_size < 0:
        raise ValueError("Gallery size must be non-negative.")
    if (
        input_dir == output_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError("Input and output directories must not overlap.")
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError(f"Output directory must be empty or new: {output_dir}")
    paths = sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".png"),
        key=lambda p: p.relative_to(input_dir).as_posix(),
    )
    if not paths:
        raise ValueError(f"No PNG images found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(paths):
        relative = source.relative_to(input_dir)
        destination = Path("images") / relative
        grayscale = read_grayscale(source)
        rgb = apply_viridis(grayscale)
        save_rgb(output_dir / destination, rgb)
        comparison = None
        if index < gallery_size:
            comparison = Path("comparisons") / relative
            save_comparison(output_dir / comparison, grayscale, rgb)
        records.append(
            {
                "source": relative.as_posix(),
                "output": destination.as_posix(),
                "height": grayscale.shape[0],
                "width": grayscale.shape[1],
                "comparison": comparison.as_posix() if comparison else None,
            }
        )
        LOGGER.info("Processed %s (%d/%d)", relative, index + 1, len(paths))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "method": "OpenCV COLORMAP_VIRIDIS",
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
                "input_directory": str(input_dir),
                "input_dtype": "uint8",
                "intensity_range": [0, 255],
                "normalization": "none",
                "grayscale_decoding": (
                    "native grayscale or identical RGB channels; alpha must be fully opaque"
                ),
                "output_color_order": "RGB",
                "output_format": "PNG",
                "image_count": len(records),
                "gallery_size": min(gallery_size, len(records)),
                "images": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Completed %d images; manifest: %s", len(records), manifest_path)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "viridis",
    )
    parser.add_argument("--gallery-size", type=int, default=10)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_baseline(args.input_dir, args.output_dir, args.gallery_size)
    except (OSError, ValueError, cv2.error) as error:
        LOGGER.error(
            "%s Run failed; any partial outputs are not a completed dataset.", error
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())