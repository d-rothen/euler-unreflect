"""CLI for running UnReflectAnything inference with euler-loading datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm

from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset
from unreflectanything._shared import (
    DEFAULT_WEIGHTS_FILENAME,
    download_configs,
    download_weights,
    get_cache_dir,
)
from unreflectanything.inference_ import inference
from unreflectanything.model_ import model as model_factory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UnReflectAnything inference on euler-loading datasets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- prepare --------------------------------------------------------
    prep = subparsers.add_parser(
        "prepare",
        help="Download model weights and configs to a local directory.",
    )
    prep.add_argument(
        "path",
        type=str,
        help="Directory to store weights/ and configs/ subdirectories.",
    )

    # -- infer (default) ------------------------------------------------
    inf = subparsers.add_parser(
        "infer",
        help="Run diffuse inference on a dataset.",
    )
    inf.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source RGB modality directory (must be ds-crawler indexed).",
    )
    inf.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the output diffuse images.",
    )
    inf.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory containing weights/ and configs/ subdirectories (as created by 'prepare'). "
             "Defaults to ~/.cache/unreflectanything.",
    )
    inf.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per forward pass (default: 4).",
    )
    inf.add_argument(
        "--brightness-threshold",
        type=float,
        default=0.8,
        help="Brightness threshold for highlight mask (0.0-1.0, default: 0.8).",
    )
    inf.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda).",
    )
    inf.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    inf.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )

    return parser


# -- prepare command ----------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    root = Path(args.path)
    weights_dir = root / "weights"
    configs_dir = root / "configs"

    weights_file = weights_dir / DEFAULT_WEIGHTS_FILENAME
    config_file = configs_dir / "pretrained_config.yaml"

    if weights_file.exists():
        print(f"Weights already present: {weights_file}")
    else:
        print(f"Downloading weights to {weights_dir} ...")
        download_weights(output_dir=weights_dir)

    if config_file.exists():
        print(f"Config already present: {config_file}")
    else:
        print(f"Downloading configs to {configs_dir} ...")
        download_configs(output_dir=configs_dir)

    print(f"Done. Assets ready at {root}")


# -- infer command ------------------------------------------------------

def cmd_infer(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the dataset using euler-loading
    dataset = MultiModalDataset(
        modalities={
            "rgb": Modality(args.source),
        },
    )
    print(f"Dataset: {len(dataset)} samples from {args.source}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Resolve cache directory (weights + configs)
    cache_dir = Path(args.cache_dir) if args.cache_dir else get_cache_dir("")
    weights_path = cache_dir / "weights" / DEFAULT_WEIGHTS_FILENAME
    config_path = cache_dir / "configs" / "pretrained_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model config not found at {config_path}.\n"
            f"Run 'euler-unreflect prepare {cache_dir}' on a machine with internet access first."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}.\n"
            f"Run 'euler-unreflect prepare {cache_dir}' on a machine with internet access first."
        )

    # Load the model once
    print(f"Loading model (device={args.device}) ...")
    mdl = model_factory(
        pretrained=True,
        weights_path=weights_path,
        config_path=config_path,
        device=args.device,
        verbose=args.verbose,
    )

    # Set up the writer to mirror the source directory structure
    writer = DatasetWriter(
        output_dir,
        name="diffuse",
        type="rgb",
        euler_train={"used_as": "target", "modality_type": "rgb"},
    )

    for batch in tqdm(loader, desc="Inference", unit="batch"):
        rgb = batch["rgb"]             # (B, 3, H, W) float32 in [0, 1]
        full_ids = batch["full_id"]    # list of str
        file_ids = batch["id"]         # list of str
        metas = batch["meta"]

        # Run inference with the pre-loaded model
        diffuse = inference(
            rgb,
            model=mdl,
            brightness_threshold=args.brightness_threshold,
            verbose=args.verbose,
        )  # (B, 3, H, W) float32 in [0, 1]

        # Save each image via DatasetWriter (preserves hierarchy)
        for i in range(diffuse.shape[0]):
            source_meta = {
                k: metas["rgb"][k][i] for k in metas["rgb"]
            }
            out_path = writer.get_path(
                full_id=full_ids[i],
                basename=f"{file_ids[i]}.png",
                source_meta=source_meta,
            )
            img = TF.to_pil_image(diffuse[i].clamp(0.0, 1.0).cpu())
            img.save(out_path)

    writer.save_index()
    print(f"Done. {len(writer)} images saved to {output_dir}")


# -- entry point --------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "infer":
        cmd_infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
