# -*- coding: utf-8 -*-
"""
Proxy-GS ablation curves: PSNR vs Proxy Resolution / Vertex Noise
- 左图：Proxy Resolution = [100%, 10%, 5%, 1%]
- 右图：Noisy Magnitude = [0%, 5%, 10%, 20%]
修改下面 METHODS 里的数值即可绘图。
"""

import matplotlib.pyplot as plt

# ============ 1) 需要你填的实验数据（示例数值，可自由修改） ============
# 每个方法包含两组列表：'resolution' 和 'noise'，长度都必须是 4
METHODS = {
    "Proxy-GS": {
        "resolution": [26.44, 26.10, 25.90, 25.40],  # 对应 [100%, 10%, 5%, 1%]
        "noise":      [26.44, 26.20, 25.80, 25.00],  # 对应 [0%, 5%, 10%, 20%]
    },
}
# 如果只想画一条或两条曲线，删除对应的方法即可；也可以改成你论文里的方法名。

# 横轴标签
X_LABELS_RES = ["100%", "10%", "5%", "1%"]
X_LABELS_NOI = ["0%", "5%", "10%", "20%"]

# ============ 2) 画图参数（可按需微调） ============
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]
LW = 2.0
MS = 6
FIGSIZE = (10, 4)  # 并排两子图
Y_LABEL = "PSNR (dB)"

# 是否统一两张图的 y 轴范围（便于对比）
USE_SHARED_YLIM = True
Y_MIN, Y_MAX = 24.5, 27.2  # 若 USE_SHARED_YLIM=True，则使用这个范围；否则自动

# 输出文件名
OUT_PNG = "proxy_ablation_psnr.png"
OUT_PDF = "proxy_ablation_psnr.pdf"

# ============ 3) 绘图 ============

def plot_panel(ax, x_labels, series_dict, title):
    x = list(range(len(x_labels)))
    for idx, (name, vals) in enumerate(series_dict.items()):
        marker = MARKERS[idx % len(MARKERS)]
        ls = LINESTYLES[idx % len(LINESTYLES)]
        ax.plot(
            x, vals, label=name,
            marker=marker, linestyle=ls, linewidth=LW, markersize=MS
        )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)

# 整理每个面板的数据
series_res = {name: d["resolution"] for name, d in METHODS.items()}
series_noi = {name: d["noise"] for name, d in METHODS.items()}

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, constrained_layout=True)

plot_panel(axes[0], X_LABELS_RES, series_res, "Proxy Resolution")
plot_panel(axes[1], X_LABELS_NOI, series_noi, "Vertex Noise")

# 统一 y 轴
if USE_SHARED_YLIM:
    axes[0].set_ylim(Y_MIN, Y_MAX)
    axes[1].set_ylim(Y_MIN, Y_MAX)

# 合并图例（放在右上角，可按需修改 loc / ncol）
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.04))

# 保存
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved to {OUT_PNG} and {OUT_PDF}")

# 如果需要在脚本里直接展示
# plt.show()
