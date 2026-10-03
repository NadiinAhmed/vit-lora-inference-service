"""Send realistic traffic to the running service, to demonstrate monitoring end to end.

Start the service first (uvicorn or Docker), then from the repository root:
    python -m monitoring.simulate_traffic --scenario normal
    python -m monitoring.simulate_traffic --scenario dark

Images come from the TrashNet VALIDATION split: never trained on, and different photos
from the test images in the reference. So "normal" traffic is similar to the reference
without being identical to it, just like real traffic from unchanged conditions.

    normal - images sent as they are                    -> report should be OK
    dark   - the same images at 45% brightness, like a   -> report should ALERT
             badly lit sorting line

Each image's true label is then sent to POST /feedback, playing the role of a person
who checks predictions (in real life only some predictions would get a label).
"""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageEnhance

from src.config import get_settings
from src.preprocessing import open_image
from training.data import download_trashnet, load_splits

DARK_BRIGHTNESS = 0.45


def darken(image: Image.Image, factor: float = DARK_BRIGHTNESS) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def to_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def post_image(base_url: str, data: bytes, filename: str) -> dict:
    """POST /predict as multipart/form-data, using only the standard library."""
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: image/jpeg\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(f"{base_url}/predict", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def post_feedback(base_url: str, request_id: str, true_label: str) -> dict:
    body = json.dumps({"request_id": request_id, "true_label": true_label}).encode()
    request = urllib.request.Request(f"{base_url}/feedback", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=["normal", "dark"], default="normal")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--limit", type=int, default=None, help="send at most N images")
    parser.add_argument("--feedback-rate", type=float, default=1.0,
                        help="share of predictions that receive a true label (0 = no labels at all)")
    parser.add_argument("--data-root", type=Path, default=None, help="TrashNet class folders (default: download)")
    args = parser.parse_args()
    settings = get_settings()

    summary = json.loads((settings.reports_dir / "training_summary.json").read_text())
    data_root = args.data_root or download_trashnet(settings.data_dir)
    splits = load_splits(data_root, summary["val_fraction"], summary["test_fraction"], summary["seed"])
    samples = list(splits.validation)
    random.Random(0).shuffle(samples)  # mix the classes (the split lists them class by class), then limit
    samples = samples[:args.limit]
    rng = random.Random(1)

    print(f"Sending {len(samples)} validation images to {args.url} (scenario: {args.scenario}) ...")
    correct = labelled = 0
    for index, (path, label_id) in enumerate(samples, start=1):
        image = open_image(path)
        if args.scenario == "dark":
            image = darken(image)
        prediction = post_image(args.url, to_jpeg(image), path.name)

        if rng.random() < args.feedback_rate:
            result = post_feedback(args.url, prediction["request_id"], splits.class_names[label_id])
            labelled += 1
            correct += result["correct"]
        if index % 50 == 0:
            print(f"  {index}/{len(samples)} sent")

    accuracy = f"{correct / labelled:.4f}" if labelled else "n/a"
    print(f"Done: {len(samples)} predictions, {labelled} with feedback, accuracy on labelled = {accuracy}")
    print(f"Next: python -m monitoring.report --last {len(samples)}")


if __name__ == "__main__":
    main()
