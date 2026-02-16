# euler-unreflect

CLI for running [UnReflectAnything](https://github.com/d-rothen/unreflectanything) inference on [euler-loading](https://github.com/d-rothen/euler-loading) datasets.

## Installation

```bash
uv pip install "euler-unreflect @ git+https://github.com/d-rothen/euler-unreflect.git"
```

## Usage

```bash
python -m euler_unreflect \
    --source /data/my_dataset/rgb \
    --output /data/my_dataset/diffuse \
    --weights /path/to/weights.pt \
    --batch-size 8 \
    --brightness-threshold 0.8
```

The `--source` directory must be a ds-crawler indexed modality (i.e. contain a `ds-crawler.config` or cached `output.json`).

Output images are written in the same directory structure as the source, and a `.ds_crawler/output.json` index is generated so the output can be used directly as an euler-loading modality.

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--source` | yes | | Path to source RGB modality directory |
| `--output` | yes | | Directory for output diffuse images |
| `--weights` | no | cached default | Path to model weights checkpoint |
| `--batch-size` | no | `4` | Images per forward pass |
| `--brightness-threshold` | no | `0.8` | Highlight mask threshold (0.0 - 1.0) |
| `--device` | no | `cuda` | Inference device (`cuda`, `cpu`) |
| `--num-workers` | no | `4` | DataLoader worker count |
| `--verbose` | no | off | Print progress |
