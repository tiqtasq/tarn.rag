"""Fetch the ONNX embedding model + tokenizer into MODEL_DIR (one-time, needs network).

After this, ingestion and retrieval run fully offline against the local files. The defaults
match the project default model; override via args/env to swap models (keep ingestion and
retrieval pointed at the same one — the fingerprint enforces it).

    python scripts/fetch_model.py [--repo HF_REPO] [--model-file onnx/model.onnx] [--dest DIR]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Xenova/gte-small")  # ONNX export of thenlper/gte-small (the default)
    ap.add_argument("--model-file", default="onnx/model.onnx")
    ap.add_argument("--tokenizer-file", default="tokenizer.json")
    ap.add_argument("--dest", default="./models/gte-small")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for src_name, out_name in [(args.model_file, "model.onnx"), (args.tokenizer_file, "tokenizer.json")]:
        cached = hf_hub_download(args.repo, src_name)
        shutil.copyfile(cached, dest / out_name)
        print(f"  {src_name} -> {dest / out_name}")
    print(f"Model ready in {dest}")


if __name__ == "__main__":
    main()
