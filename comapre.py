import json

# 读取两个 JSON 文件
with open("ouputs/berlin-ours_new/per_view.json", "r") as f1, open("output_octree/belin_Octree/per_view.json", "r") as f2:
    ours = json.load(f1)
    octree = json.load(f2)

# 提取 PSNR 字典
psnr_ours = ours["ours_40000"]["PSNR"]
psnr_octree = octree["ours_40000"]["PSNR"]

# 存放 (差值, 图像名, ours 值, octree 值)
diffs = []

for img_id, val_ours in psnr_ours.items():
    if img_id in psnr_octree:
        diff = val_ours - psnr_octree[img_id]
        diffs.append((diff, img_id, val_ours, psnr_octree[img_id]))

# 按差值降序排序
diffs.sort(key=lambda x: x[0], reverse=True)

# 输出前 10 个
print("前十个差值最大的图像:")
for i, (diff, img_id, ours_val, octree_val) in enumerate(diffs[:10], 1):
    print(f"{i}. 图像: {img_id}, berlin-ours_new PSNR: {ours_val:.4f}, belin_Octree PSNR: {octree_val:.4f}, 差值: {diff:.4f}")
