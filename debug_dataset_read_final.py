# debug_dataset_read_final2.py
from dataset.vos_dataset import EventbaseVOSMergeTrainDataset
from dataset.load_subset import load_subset, load_empty_masks
import torch

# 配置路径
DATA_ROOT = r"D:\download\lowlight_event_vos"

# LLE-VOS 配置
image_dir = f"{DATA_ROOT}/Lowlight_images"
mask_dir = f"{DATA_ROOT}/Annotations"
event_dir = f"{DATA_ROOT}/EventVoxel"

subset = load_subset(None)  # 全部序列
empty_masks = load_empty_masks(None)  # 默认空字典

# 创建 Dataset
dataset = EventbaseVOSMergeTrainDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    event_dir=event_dir,
    seq_length=8,
    num_ref_frames=2,
    crop_size=(480, 480),
    max_crop_trials=1,
    subset=subset,
    empty_masks=empty_masks
)

# 读取前 3 个样本
for i in range(min(3, len(dataset))):
    sample = dataset[i]
    print(f"Sample {i}:")
    print("  Keys:", sample.keys())
    print("  RGB shape:", sample['rgb'].shape)
    print("  Event voxel shape:", sample['events'].shape)
    print("  First frame mask shape:", sample['first_frame_gt'].shape)
    print("  ----------------------------")