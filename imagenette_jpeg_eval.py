"""Evaluate JPEG robustness of pretrained ResNet-18 and ViT-B/16 on Kaggle GPU.

Enable GPU and Internet in Kaggle, then run: python imagenette_jpeg_eval.py
Use --val-dir to point to an already mounted Imagenette 320 px val directory.
Saves Top-1 scores, all correct-to-wrong flips, and a ZIP of visual comparisons.
"""

import argparse
import csv
from dataclasses import dataclass
from io import BytesIO
from itertools import groupby
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
from tempfile import TemporaryDirectory, gettempdir
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16
from torchvision.transforms.functional import center_crop, resize


DATASET = "imagenette2-320"
DATA_URL = f"https://s3.amazonaws.com/fast-ai-imageclas/{DATASET}.tgz"
ARCHIVE_MD5 = "3df6f0d01a2c9592104656642f5e78a3"
CACHE_DIR = Path(gettempdir()) / "imagenette_cache"
VALIDATION_COUNT = 3925
QUALITIES = (None, 95, 75, 50, 25, 10)

# WordNet directory names mapped to zero-based ImageNet-1K output indices.
IMAGENET_INDEX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
MODEL_SPECS = (
    ("resnet18", resnet18, ResNet18_Weights.IMAGENET1K_V1),
    ("vit_b_16", vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
)
FLIP_COLUMNS = (
    "model", "jpeg_quality", "image_path", "true_class", "true_index",
    "baseline_prediction", "baseline_index", "jpeg_prediction", "jpeg_index",
)


def download_validation_set(cache_dir):
    val_dir = cache_dir / DATASET / "val"
    if val_dir.is_dir():
        return val_dir

    val_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as temporary:
        staging = Path(temporary)
        archive = staging / f"{DATASET}.tgz"
        print(f"Downloading {DATASET} (about 342 MB)...", flush=True)
        download_url(DATA_URL, str(staging), filename=archive.name, md5=ARCHIVE_MD5)

        extracted_val = staging / "val"
        print("Extracting the validation split...", flush=True)
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                path = PurePosixPath(member.name)
                if (
                    len(path.parts) == 4
                    and path.parts[:2] == (DATASET, "val")
                    and path.parts[2] in IMAGENET_INDEX
                    and path.suffix.lower() in {".jpg", ".jpeg"}
                    and member.isfile()
                ):
                    destination = extracted_val / path.parts[2] / path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as image_file, destination.open("wb") as file:
                        shutil.copyfileobj(image_file, file)

        load_validation_set(extracted_val)
        extracted_val.rename(val_dir)
    return val_dir


def load_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def load_validation_set(val_dir):
    dataset = ImageFolder(val_dir, loader=load_rgb)
    if set(dataset.classes) != IMAGENET_INDEX.keys() or len(dataset) != VALIDATION_COUNT:
        raise ValueError(f"Expected all 10 Imagenette classes and {VALIDATION_COUNT} validation images in {val_dir}.")
    return dataset


def reencode_jpeg(image, quality):
    # Encode at the dataset resolution, with the same settings as the preview.
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling="4:2:0")
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


@dataclass
class JPEGView:
    quality: int | None
    preprocess: torch.nn.Module

    def __call__(self, image):
        if self.quality is not None:
            image = reencode_jpeg(image, self.quality)
        return self.preprocess(image)


@torch.inference_mode()
def evaluate(dataset, models, device, batch_size, workers):
    class_indices = [IMAGENET_INDEX[wnid] for wnid in dataset.classes]
    targets = torch.tensor([class_indices[label] for label in dataset.targets])
    total = len(targets)
    # Both selected weight sets specify the same 256 resize, 224 center crop and normalization.
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rows = []
    flips = []
    baseline_correct = {}
    for quality in QUALITIES:
        dataset.transform = JPEGView(quality, preprocess)
        batches = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=device.type == "cuda",
        )
        condition = "baseline" if quality is None else "jpeg"
        label = "baseline" if quality is None else f"JPEG Q={quality}"
        print(f"Evaluating {label}...", flush=True)
        predictions = {name: [] for name, _, _ in models}
        for images, _ in batches:
            images = images.to(device, non_blocking=True)
            for name, _, model in models:
                predictions[name].append(model(images).argmax(dim=1))

        for name, weights, _ in models:
            predicted = torch.cat(predictions[name]).cpu()
            correct = predicted == targets
            count = int(correct.sum())
            percent = 100.0 * count / total
            if quality is None:
                baseline_correct[name] = correct
            else:
                indices = (baseline_correct[name] & ~correct).nonzero(as_tuple=True)[0].tolist()
                categories = weights.meta["categories"]
                for index in indices:
                    true_index = int(targets[index])
                    jpeg_index = int(predicted[index])
                    path = Path(dataset.samples[index][0]).relative_to(dataset.root).as_posix()
                    flips.append({
                        "model": name,
                        "jpeg_quality": quality,
                        "image_path": path,
                        "true_class": categories[true_index],
                        "true_index": true_index,
                        "baseline_prediction": categories[true_index],
                        "baseline_index": true_index,
                        "jpeg_prediction": categories[jpeg_index],
                        "jpeg_index": jpeg_index,
                    })
            baseline_count = int(baseline_correct[name].sum())
            rows.append({
                "model": name,
                "weights": weights.name,
                "condition": condition,
                "jpeg_quality": quality,
                "n_images": total,
                "n_correct": count,
                "top1_percent": round(percent, 6),
                "delta_vs_baseline_pp": round(100.0 * (count - baseline_count) / total, 6),
            })
            print(f"  {name}: {percent:.2f}% ({count}/{total})", flush=True)
    return rows, flips


def save_csv(output, rows, columns):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_flip_sheet(val_dir, rows, preprocess, title):
    tile, margin, header, row_height = 224, 20, 58, 294
    right = margin + tile + 32
    sheet = Image.new("RGB", (right + tile + margin, header + row_height * len(rows)), "#f5f7fa")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    ink = "#182433"
    draw.text((margin, 18), title, fill=ink, font=ImageFont.load_default(size=18))
    for row_number, row in enumerate(rows):
        y = header + row_number * row_height
        source = load_rgb(val_dir / row["image_path"])
        compressed = reencode_jpeg(source, row["jpeg_quality"])
        draw.text((margin, y), f"True: {row['true_class']}", fill=ink, font=font)
        draw.text((margin, y + 19), row["image_path"], fill=ink, font=font)
        for x, image in ((margin, source), (right, compressed)):
            image = resize(image, preprocess.resize_size, preprocess.interpolation)
            sheet.paste(center_crop(image, preprocess.crop_size), (x, y + 38))
        draw.text((margin, y + 266), f"Baseline: {row['baseline_prediction']}", fill="#16634d", font=font)
        draw.text((right, y + 266), f"JPEG: {row['jpeg_prediction']}", fill="#a32926", font=font)
    return sheet


def save_flip_previews(output, val_dir, flips, examples_per_condition):
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    rng = random.Random(42)
    with ZipFile(output, "w") as archive:
        if not flips:
            archive.writestr("README.txt", "No baseline-correct images became incorrect after JPEG recompression.\n")
        for (model, quality), group in groupby(flips, key=lambda row: (row["model"], row["jpeg_quality"])):
            rows = list(group)
            total = len(rows)
            if examples_per_condition and total > examples_per_condition:
                rows = rng.sample(rows, examples_per_condition)

            page_size = 5
            for start in range(0, len(rows), page_size):
                end = min(start + page_size, len(rows))
                title = f"{model} | JPEG Q={quality} | {start + 1}-{end} of {len(rows)} shown ({total} flips)"
                sheet = make_flip_sheet(val_dir, rows[start:end], preprocess, title)
                with BytesIO() as buffer:
                    sheet.save(buffer, format="PNG")
                    archive.writestr(f"{model}_q{quality:02d}_{start // page_size + 1:04d}.png", buffer.getvalue())


def main():
    work_dir = Path("/kaggle/working")
    if not work_dir.is_dir():
        work_dir = Path.cwd()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, help="Existing Imagenette 320 px validation directory.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2, help="CPU workers for JPEG and preprocessing.")
    parser.add_argument("--examples-per-condition", type=int, default=5,
                        help="Visual examples per model and JPEG quality. Use 0 to render every flip.")
    parser.add_argument("--output", type=Path, default=work_dir / "imagenette_jpeg_top1.csv")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers must be nonnegative.")
    if args.examples_per_condition < 0:
        parser.error("--examples-per-condition must be nonnegative.")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Enable a GPU accelerator in Kaggle and rerun the script.")

    try:
        val_dir = args.val_dir or download_validation_set(CACHE_DIR)
        dataset = load_validation_set(val_dir)
    except (OSError, RuntimeError, tarfile.TarError, ValueError) as error:
        parser.exit(1, f"Could not load Imagenette: {error}\nEnable Kaggle Internet or use --val-dir.\n")

    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(device)} for {len(dataset):,} validation images.", flush=True)
    models = [(name, weights, build(weights=weights).eval().to(device)) for name, build, weights in MODEL_SPECS]
    rows, flips = evaluate(dataset, models, device, args.batch_size, args.workers)
    flip_csv = args.output.with_name(f"{args.output.stem}_flips.csv")
    previews = args.output.with_name(f"{args.output.stem}_flip_previews.zip")
    save_csv(args.output, rows, list(rows[0]))
    save_csv(flip_csv, flips, FLIP_COLUMNS)
    print(f"Saved {len(rows)} results to {args.output.resolve()}")
    print(f"Saved {len(flips)} flip records to {flip_csv.resolve()}", flush=True)
    save_flip_previews(previews, val_dir, flips, args.examples_per_condition)
    print(f"Saved flip previews to {previews.resolve()}")


if __name__ == "__main__":
    main()
