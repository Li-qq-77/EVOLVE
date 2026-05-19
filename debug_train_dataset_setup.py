import os

from hydra import initialize, compose
from omegaconf import OmegaConf

os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

from dataset.setup_training_data import setup_main_training_event_dataset


def main():
    with initialize(version_base="1.3.2", config_path="config"):
        cfg = compose(
            config_name="train_config.yaml",
            overrides=[
                "data=llevos_debug",
                "model=small",
                "pre_training.enabled=False",
                "main_training.enabled=True",
                "main_training.batch_size=1",
                "main_training.crop_size=[240,240]",
                "main_training.seq_length=3",
                "num_workers=0",
            ],
        )

    print("[CONFIG OK]")
    print(OmegaConf.to_yaml(cfg.data))

    dataset, sampler, loader = setup_main_training_event_dataset(cfg, max_skip=5)

    print("[DATASET SETUP OK]")
    print("dataset length =", len(dataset))
    print("loader length =", len(loader))

    sample = dataset[0]
    print("[SAMPLE OK]")
    print("sample keys =", sample.keys())
    print("rgb shape =", sample["rgb"].shape)
    print("events shape =", sample["events"].shape)
    print("first_frame_gt shape =", sample["first_frame_gt"].shape)
    print("cls_gt shape =", sample["cls_gt"].shape)
    print("selector shape =", sample["selector"].shape)
    print("info =", sample["info"])

    batch = next(iter(loader))
    print("[LOADER OK]")
    print("batch keys =", batch.keys())
    print("batch rgb shape =", batch["rgb"].shape)
    print("batch events shape =", batch["events"].shape)
    print("batch first_frame_gt shape =", batch["first_frame_gt"].shape)
    print("batch cls_gt shape =", batch["cls_gt"].shape)
    print("batch selector shape =", batch["selector"].shape)


if __name__ == "__main__":
    main()
