# save as: convert_transforms.py
import json
from pathlib import Path
import numpy as np

in_path  = "data/small_city/street/pose/block_small/transforms_origin.json"
out_path = "data/small_city/street/pose/block_small/transforms_converted.json"

# 目标路径前缀（四位零填充）
prefix = "../../train/small_city_road_horizon"

with open(in_path, "r") as f:
    meta = json.load(f)

new_frames = []
for i, fr in enumerate(meta.get("frames", [])):
    # 1) 取 index（优先 frame_index，缺省用枚举 i）
    idx = fr.get("frame_index", i)
    idx_str = f"{idx:04d}"

    # 2) 设定 file_path
    fr["file_path"] = f"{prefix}/{idx_str}.png"

    # 3) 取矩阵（rot_mat -> transform_matrix）
    if "rot_mat" in fr:
        mat = np.array(fr["rot_mat"], dtype=float)
        del fr["rot_mat"]
    else:
        mat = np.array(fr["transform_matrix"], dtype=float)

    if mat.shape != (4, 4):
        raise ValueError(f"frame {idx}: 矩阵不是 4x4，实际 {mat.shape}")

    # 4) 按要求缩放：旋转等“其它部分”×100，平移列 ÷100；最后一行保持 [0,0,0,1]
    mat[:3, :3] *= 100.0     # 其它部分（3x3）
    mat[:3,  3] /= 100.0     # 平移列
    mat[3, :]   = [0.0, 0.0, 0.0, 1.0]  # 保持齐次行

    fr["transform_matrix"] = mat.tolist()

    # 5) 删除 frame_index（如需保留可注释掉）
    if "frame_index" in fr:
        del fr["frame_index"]

    new_frames.append(fr)

# 如需按编号排序，取消下面两行的注释
# new_frames.sort(key=lambda x: int(Path(x["file_path"]).stem))
meta["frames"] = new_frames

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"done: {out_path}")
