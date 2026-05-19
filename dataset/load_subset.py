from collections import defaultdict


def load_subset(file_path):
    """
    读取 subset 文件。
    文件格式：每行一个视频序列名。
    """
    if file_path is None:
        return None

    subset = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            subset.add(line.split()[0])

    return subset


def load_empty_masks(file_path):
    """
    读取空 mask 文件。
    支持格式：
    1. video_name frame_name
    2. video_name/frame_name
    3. video_name/frame_name.png
    """
    if file_path is None:
        return None

    empty_masks = defaultdict(set)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace("\\", "/").split()

            if len(parts) >= 2:
                video, frame = parts[0], parts[1]
            else:
                item = parts[0]
                if "/" not in item:
                    continue
                video, frame = item.rsplit("/", 1)

            if "." in frame:
                frame = frame.rsplit(".", 1)[0]

            empty_masks[video].add(frame)

    return dict(empty_masks)
