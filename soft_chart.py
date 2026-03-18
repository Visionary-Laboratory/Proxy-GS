import matplotlib.pyplot as plt

# GS 数量 (横坐标)，转为整数
gs_numbers = [356, 178, 118, 89, 59]

# FPS 数据
soft_fps = [31.69, 58.32, 86.25, 112.79, 158.38]
hard_fps = [30.51, 68.76, 103.466, 135.01, 199.034]

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(gs_numbers, soft_fps, marker='o', label="Soft Rasterization")
plt.plot(gs_numbers, hard_fps, marker='s', label="Hard Rasterization")

# 标注数据点
for x, y in zip(gs_numbers, soft_fps):
    plt.text(x, y + 3, f"{y:.1f}", ha='center', fontsize=8, color='orange')
for x, y in zip(gs_numbers, hard_fps):
    plt.text(x, y + 3, f"{y:.1f}", ha='center', fontsize=8, color='blue')

# 设置坐标轴和标题
plt.xlabel("GS Number (w)")
plt.ylabel("FPS")
plt.title("Soft vs Hard Rasterization)")
plt.legend()
plt.grid(True)

# 翻转x轴（GS减少时FPS增加）
plt.gca().invert_xaxis()

# 保存图片
plt.savefig("fps_vs_gs.pdf", dpi=300, bbox_inches="tight")

print("图像已保存为 fps_vs_gs.pdf")
