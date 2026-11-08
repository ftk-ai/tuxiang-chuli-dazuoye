"""Compare JPEG quality on Imagenette 320 px validation images (Pillow 10.1+).

Run with: python imagenette_jpeg_preview.py
On Kaggle, enable Internet for the first run, or pass --val-dir for a mounted copy.
"""

import argparse
from io import BytesIO
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
from tempfile import TemporaryDirectory, gettempdir
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont, ImageOps


DATASET = "imagenette2-320"
DATA_URL = f"https://s3.amazonaws.com/fast-ai-imageclas/{DATASET}.tgz"
WORK_DIR = Path("/kaggle/working")
CACHE_DIR = Path(gettempdir()) / "imagenette_cache"
if not WORK_DIR.is_dir():
    WORK_DIR = Path.cwd()

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def download_validation_set(cache_dir):
    val_dir = cache_dir / DATASET / "val"
    if val_dir.is_dir():
        return val_dir

    val_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as temporary:
        temporary = Path(temporary)
        archive = temporary / f"{DATASET}.tgz"
        print(f"Downloading {DATASET} (about 342 MB)...", flush=True)
        with urlopen(DATA_URL, timeout=60) as response, archive.open("wb") as file:
            shutil.copyfileobj(response, file, length=1024 * 1024)

        print("Extracting only the validation split...", flush=True)
        extracted_val = temporary / DATASET / "val"
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                path = PurePosixPath(member.name)
                if (
                    len(path.parts) == 4
                    and path.parts[:2] == (DATASET, "val")
                    and path.parts[2] in CLASS_NAMES
                    and path.suffix.lower() in {".jpg", ".jpeg"}
                    and member.isfile()
                ):
                    destination = extracted_val / path.parts[2] / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as image_file, destination.open("wb") as file:
                        shutil.copyfileobj(image_file, file)

        if not extracted_val.is_dir():
            raise ValueError("The archive did not contain the expected validation split.")
        extracted_val.rename(val_dir)
    return val_dir


def sample_images(val_dir, count, seed):
    if not val_dir.is_dir():
        raise ValueError(f"Validation directory does not exist: {val_dir}")

    images_by_class = {}
    for class_dir in sorted(val_dir.iterdir()):
        if class_dir.is_dir() and class_dir.name in CLASS_NAMES:
            images = sorted(
                path for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            )
            if images:
                images_by_class[class_dir.name] = images

    if count > len(images_by_class):
        raise ValueError(f"Requested {count} classes, but found {len(images_by_class)} in {val_dir}.")

    rng = random.Random(seed)
    classes = rng.sample(list(images_by_class), count)
    print(f"Found {sum(map(len, images_by_class.values())):,} validation JPEGs in {val_dir}")
    return [rng.choice(images_by_class[class_id]) for class_id in classes]


def jpeg_round_trip(image, quality):
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling="4:2:0")
        size_bytes = buffer.tell()
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB"), size_bytes


def save_preview(paths, qualities, output):
    rows = []
    for path in paths:
        with Image.open(path) as source:
            baseline = ImageOps.exif_transpose(source).convert("RGB")
        panels = [(baseline, path.stat().st_size)]
        panels.extend(jpeg_round_trip(baseline, quality) for quality in qualities)
        rows.append((path, panels))

    margin, gap, header, row_label = 20, 12, 65, 28
    column_width = max(175, *(panels[0][0].width for _, panels in rows))
    columns = 1 + len(qualities)
    width = 2 * margin + columns * column_width + (columns - 1) * gap
    height = header + margin + sum(panels[0][0].height + row_label + 38 for _, panels in rows)
    gallery = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(gallery)
    ink = "#182433"
    title_font = ImageFont.load_default(size=21)
    label_font = ImageFont.load_default(size=15)
    draw.text((margin, margin), "Imagenette 320 px validation | JPEG quality comparison", font=title_font, fill=ink)

    labels = ["Baseline (no re-encode)", *(f"Quality {quality}" for quality in qualities)]
    y = header
    for path, panels in rows:
        baseline = panels[0][0]
        class_name = CLASS_NAMES[path.parent.name]
        draw.text((margin, y), f"{class_name}  |  {path.name}  |  {baseline.width} x {baseline.height}", font=label_font, fill=ink)
        y += row_label
        for column, (label, (image, size_bytes)) in enumerate(zip(labels, panels)):
            x = margin + column * (column_width + gap)
            draw.text((x, y), f"{label}  |  {size_bytes / 1024:.1f} KiB", font=label_font, fill=ink)
            gallery.paste(image, (x, y + 22))
        y += baseline.height + 38

    output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output, format="PNG")
    print(f"Preview saved to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, help="Path to an existing Imagenette 320 px val directory.")
    parser.add_argument("--qualities", type=int, nargs="+", default=[95, 75, 50, 25, 10])
    parser.add_argument("--samples", type=int, default=3, help="Number of distinct classes to preview.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=WORK_DIR / "imagenette_jpeg_preview.png")
    args = parser.parse_args()
    if any(not 1 <= quality <= 100 for quality in args.qualities):
        parser.error("JPEG quality must be in the range 1 to 100.")
    if not 1 <= args.samples <= len(CLASS_NAMES):
        parser.error(f"--samples must be in the range 1 to {len(CLASS_NAMES)}.")

    try:
        val_dir = args.val_dir or download_validation_set(CACHE_DIR)
        paths = sample_images(val_dir, args.samples, args.seed)
    except URLError as error:
        parser.exit(1, f"Download failed: {error}. Enable Kaggle Internet or pass --val-dir.\n")
    except (OSError, tarfile.TarError, ValueError) as error:
        parser.exit(1, f"Could not load Imagenette: {error}\n")

    save_preview(paths, args.qualities, args.output)


if __name__ == "__main__":
    main()
