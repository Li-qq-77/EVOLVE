import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

# Make local debug run behave like a single-process training job.
os.environ.setdefault("LOCAL_RANK", "0")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_cfg(model_name: str, no_pretrained_backbone: bool):
    base_model = OmegaConf.load(PROJECT_ROOT / "config" / "model" / "base.yaml")
    if model_name == "small":
        small_model = OmegaConf.load(PROJECT_ROOT / "config" / "model" / "small.yaml")
        if "defaults" in small_model:
            small_model.pop("defaults")
        model_cfg = OmegaConf.merge(base_model, small_model)
    elif model_name == "base":
        model_cfg = base_model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if no_pretrained_backbone:
        # Avoid internet downloads during local interface/shape debugging.
        model_cfg.resnet_pretrained = False

    OmegaConf.resolve(model_cfg)
    return OmegaConf.create({"model": model_cfg})


def build_stage_cfg(args):
    return OmegaConf.create({
        "name": "debug_forward",
        "seq_length": args.seq_length,
        "num_ref_frames": min(args.num_ref_frames, max(args.seq_length - 1, 1)),
        "num_objects": args.max_num_obj,
        "deep_update_prob": 0.0,
        "amp": bool(args.amp and args.device.startswith("cuda")),
    })


def build_dataset(args):
    from dataset.vos_dataset import EventbaseVOSMergeTrainDataset

    data_root = Path(args.data_root)
    image_root = data_root / args.image_dir
    mask_root = data_root / args.mask_dir
    event_root = data_root / args.event_dir
    missing = [str(p) for p in (image_root, mask_root, event_root) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(missing))

    data_configs = {
        "LLEVOS": {
            "im_root": str(image_root),
            "gt_root": str(mask_root),
            "event_root": str(event_root),
            "max_skip": args.max_skip,
            "subset": None,
            "empty_masks": None,
            "multiplier": 1,
        }
    }
    return EventbaseVOSMergeTrainDataset(
        data_configs=data_configs,
        seq_length=args.seq_length,
        max_num_obj=args.max_num_obj,
        size=args.size,
        merge_probability=0.0,
    )


def as_tensor(x: Any, *, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.contiguous()


def collate_one_sample(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    info = sample.get("info", {})
    selector = as_tensor(sample["selector"], dtype=torch.float32)
    num_objects = int(info.get("num_objects", selector.sum().item()))

    return {
        "rgb": as_tensor(sample["rgb"], dtype=torch.float32).unsqueeze(0).to(device),
        "events": as_tensor(sample["events"], dtype=torch.float32).unsqueeze(0).to(device),
        "first_frame_gt": as_tensor(sample["first_frame_gt"], dtype=torch.float32).unsqueeze(0).to(device),
        "cls_gt": as_tensor(sample["cls_gt"], dtype=torch.long).unsqueeze(0).to(device),
        "selector": selector.unsqueeze(0).to(device),
        "info": {
            "name": [info.get("name", "unknown")],
            "frames": [info.get("frames", [])],
            "num_objects": torch.tensor([num_objects], dtype=torch.long, device=device),
        },
    }


def describe_tree(prefix: str, obj: Any):
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}: tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}")
    elif isinstance(obj, dict):
        print(f"{prefix}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe_tree(f"{prefix}.{k}", v)
    else:
        print(f"{prefix}: {type(obj).__name__} = {obj}")


def load_optional_weights(model: torch.nn.Module, weights: Optional[str]):
    if not weights:
        print("      weights = None; using random initialization for interface test")
        return
    weight_path = Path(weights)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu")
    state = ckpt.get("weights", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    if hasattr(model, "load_weights"):
        model.load_weights(state)
    else:
        model.load_state_dict(state, strict=False)
    print(f"      weights = {weight_path}")


def main():
    parser = argparse.ArgumentParser(description="EVOLVE/CUTIE local minimum forward test")
    parser.add_argument("--data-root", default=r"D:\download\lowlight_event_vos")
    parser.add_argument("--image-dir", default="Lowlight_images")
    parser.add_argument("--mask-dir", default="Annotations")
    parser.add_argument("--event-dir", default="EventVoxel")
    parser.add_argument("--model", choices=["small", "base"], default="small")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--num-ref-frames", type=int, default=3)
    parser.add_argument("--max-num-obj", type=int, default=3)
    parser.add_argument("--size", type=int, default=480)
    parser.add_argument("--max-skip", type=int, default=5)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--use-pretrained-backbone",
        action="store_true",
        help="Allow ResNet weight loading/downloading. Disabled by default for local debugging.",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    print("[0/5] Dataset paths")
    print(f"      data_root = {Path(args.data_root)}")
    print(f"      image_dir = {args.image_dir}")
    print(f"      mask_dir  = {args.mask_dir}")
    print(f"      event_dir = {args.event_dir}")

    print("[1/5] Build dataset")
    dataset = build_dataset(args)
    print(f"      dataset length = {len(dataset)}")

    print("[2/5] Read one sample and build batch")
    sample = dataset[args.sample_index]
    data = collate_one_sample(sample, device)
    describe_tree("data.rgb", data["rgb"])
    describe_tree("data.events", data["events"])
    describe_tree("data.first_frame_gt", data["first_frame_gt"])
    describe_tree("data.cls_gt", data["cls_gt"])
    describe_tree("data.selector", data["selector"])
    describe_tree("data.info", data["info"])

    assert data["rgb"].dim() == 5, "rgb should be [B,T,3,H,W]"
    assert data["events"].dim() == 5 and data["events"].shape[2] == 5, "events should be [B,T,5,H,W]"
    assert data["first_frame_gt"].dim() == 5, "first_frame_gt should be [B,1,N,H,W]"

    print("[3/5] Build model config")
    cfg = build_cfg(args.model, no_pretrained_backbone=not args.use_pretrained_backbone)
    stage_cfg = build_stage_cfg(args)

    print("[4/5] Initialize CutieTrainWrapper")
    from cutie.model.train_wrapper import CutieTrainWrapper

    model = CutieTrainWrapper(cfg, stage_cfg).to(device)
    load_optional_weights(model, args.weights)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      model={args.model}, params={total_params:,}, device={device}")

    print("[5/5] Run forward")
    with torch.inference_mode():
        out = model(data)

    print("[SUCCESS] Minimum forward test passed: dataset output can feed EVOLVE/CUTIE model.")
    for k in sorted(out.keys()):
        describe_tree(f"out.{k}", out[k])


if __name__ == "__main__":
    main()
