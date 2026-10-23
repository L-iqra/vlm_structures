"""
Zero-shot inference against any OpenAI-compatible vision endpoint.

The script does not start or download a model. Point --base_url at a server
that is already running (vLLM, an OpenAI-compatible host, or another provider)
and pass the model name that server expects.

Images are read from PEER-style numpy splits:

    <data_dir>/task<id>/task<id>_X_<split>.npy
    <data_dir>/task<id>/task<id>_y_<split>.npy

Example:
    python vlm_infer.py \
      --task_id 3 \
      --split test \
      --data_dir ./data \
      --base_url http://localhost:8000/v1 \
      --model my-model-name \
      --api_key "$VLM_API_KEY" \
      --output_dir ./predictions \
      --workers 8
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vlm_tasks import classify_parse_status, get_task  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot VLM inference through a hosted OpenAI-compatible endpoint"
    )
    parser.add_argument("--task_id", type=int, default=3, choices=[3, 8])
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory that contains task<id>/task<id>_X_<split>.npy and the matching label file",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="OpenAI-compatible base URL, for example http://localhost:8000/v1",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name sent in the request. Use whatever name the endpoint expects.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("VLM_API_KEY", "EMPTY"),
        help="Bearer token. Defaults to the VLM_API_KEY environment variable, or EMPTY.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory for the JSONL predictions and run metadata",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument(
        "--extra_body",
        type=str,
        default=None,
        help="Optional JSON object merged into the chat request, for provider-specific fields",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing predictions and start the run from scratch",
    )
    return parser.parse_args()


def model_slug(model_name):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    return slug or "model"


def load_task_data(data_dir, task_id, split):
    task_dir = os.path.join(data_dir, f"task{task_id}")
    x_path = os.path.join(task_dir, f"task{task_id}_X_{split}.npy")
    y_path = os.path.join(task_dir, f"task{task_id}_y_{split}.npy")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"Expected {x_path} and {y_path}. Pass the directory that holds the task folders."
        )
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path)
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    return x, y.astype(int)


def _to_uint8(image):
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.min() < 0.0 or arr.max() > 1.0:
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255).astype(np.uint8)
    return arr


def image_to_data_url(image):
    arr = _to_uint8(image)
    pil_image = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
        return " ".join(chunks)
    return str(content)


def build_request_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def read_completed_indices(path):
    """Return indices already present in a partially written JSONL run."""
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get("index"), int):
                completed.add(row["index"])
    return completed


def request_prediction(session, args, task, extra_body, headers, data_url):
    payload = {
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": task.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    payload.update(extra_body)

    url = f"{args.base_url.rstrip('/')}/chat/completions"
    last_error = None
    for attempt in range(args.max_retries):
        try:
            started = time.time()
            response = session.post(
                url, headers=headers, data=json.dumps(payload), timeout=args.timeout
            )
            response.raise_for_status()
            latency = time.time() - started
            body = response.json()
            return extract_text(body["choices"][0]["message"]["content"]), latency, None
        except Exception as exc:  # noqa: BLE001 - retry on any transport/server error
            last_error = exc
            if attempt < args.max_retries - 1:
                time.sleep(min(2**attempt, 15))
    return "", 0.0, last_error


def run_inference(args):
    task = get_task(args.task_id)
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    if not isinstance(extra_body, dict):
        raise ValueError("--extra_body must be a JSON object")

    slug = model_slug(args.model)
    headers = build_request_headers(args.api_key)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"task{args.task_id}_{slug}_{args.split}.jsonl")

    print(f"Task:            {task.title}")
    print(f"Model:           {args.model}")
    print(f"Endpoint:        {args.base_url}")
    print(f"Data directory:  {args.data_dir}")
    print(f"Output:          {output_path}")
    print(f"Workers:         {args.workers}")
    print(f"API key:         {'set' if args.api_key else 'not set'}")

    x, y = load_task_data(args.data_dir, args.task_id, args.split)
    total = len(x) if args.max_samples is None else min(len(x), args.max_samples)

    if args.overwrite and os.path.exists(output_path):
        os.remove(output_path)
    completed = read_completed_indices(output_path)
    pending = [i for i in range(total) if i not in completed]
    if completed:
        print(f"Resuming: {len(completed)} samples already present, {len(pending)} remaining")

    write_lock = threading.Lock()
    counter = {"done": 0, "errors": 0}
    session = requests.Session()

    def process(idx):
        data_url = image_to_data_url(x[idx])
        raw_response, latency, error = request_prediction(
            session, args, task, extra_body, headers, data_url
        )
        pred_label, pred_name = task.parser(raw_response)
        parse_status = "error" if error is not None else classify_parse_status(
            raw_response, pred_label
        )
        true_label = int(y[idx])
        result = {
            "index": idx,
            "image_path": f"task{args.task_id}_X_{args.split}[{idx}]",
            "model": args.model,
            "true_label": true_label,
            "true_label_name": task.name_for(true_label),
            "pred_label": pred_label,
            "pred_label_name": pred_name,
            "parse_status": parse_status,
            "latency_s": round(latency, 4),
            "raw_response": raw_response,
        }
        if error is not None:
            result["error"] = f"{type(error).__name__}: {error}"

        with write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            counter["done"] += 1
            if error is not None:
                counter["errors"] += 1
            done = counter["done"]
            if done % 25 == 0 or done == len(pending):
                print(
                    f"[{done}/{len(pending)}] last: true={result['true_label_name']} "
                    f"pred={pred_name} status={parse_status}"
                )

    wall_start = time.time()
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(process, pending))
    wall = time.time() - wall_start

    print(f"Saved predictions: {output_path}")
    print(f"Processed samples: {len(pending)} (total in file: {len(completed) + len(pending)})")
    print(f"Request errors:    {counter['errors']}")
    print(f"Wall clock:        {wall:.1f}s")

    meta_path = os.path.join(args.output_dir, f"task{args.task_id}_{slug}_{args.split}_runmeta.json")
    meta = {
        "task_id": args.task_id,
        "split": args.split,
        "model": args.model,
        "base_url": args.base_url,
        "samples_processed": len(pending),
        "request_errors": counter["errors"],
        "wall_clock_s": round(wall, 2),
        "workers": args.workers,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved run metadata: {meta_path}")


if __name__ == "__main__":
    run_inference(parse_args())
