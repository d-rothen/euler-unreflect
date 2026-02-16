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
from unreflectanything import inference, model as model_factory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UnReflectAnything diffuse inference on an euler-loading dataset.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source RGB modality directory (must be ds-crawler indexed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the output diffuse images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights checkpoint. Uses cached default if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images per forward pass (default: 4).",
    )
    parser.add_argument(
        "--brightness-threshold",
        type=float,
        default=0.8,
        help="Brightness threshold for highlight mask (0.0-1.0, default: 0.8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

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

    # Load the model once
    print(f"Loading model (device={args.device}) ...")
    mdl = model_factory(
        pretrained=True,
        weights_path=args.weights,
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


if __name__ == "__main__":
    main()
