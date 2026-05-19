# debug_dataset_read.py
import os
from dataset.vos_dataset import EventbaseVOSMergeTrainDataset

# ------------------------
# 本地 LLE-VOS 路径配置
# ------------------------
DATA_ROOT = r"D:\download\lowlight_event_vos"
LLEVOS_config = {
    "LLEVOS": {
        "im_root": os.path.join(DATA_ROOT, "Lowlight_images"),
        "gt_root": os.path.join(DATA_ROOT, "Annotations"),
        "event_root": os.path.join(DATA_ROOT, "EventVoxel"),
        "max_skip": 5,
        "subset": None,
        "empty_masks": None,
        "multiplier": 1
    }
}

# ------------------------
# 初始化 Dataset
# ------------------------
dataset = EventbaseVOSMergeTrainDataset(
    data_configs=LLEVOS_config,
    seq_length=8,
    max_num_obj=3,
    size=480,
    merge_probability=0.0
)

# ------------------------
# 读取前 3 个样本检查
# ------------------------
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"Sample {i}:")
    print("  Keys:", sample.keys())
    print("  RGB shape:", sample['rgb'].shape)
    print("  Event voxel shape:", sample['events'].shape)
    print("  First frame mask shape:", sample['first_frame_gt'].shape)
    print("---------------------------")