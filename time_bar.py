import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Set the font family to a Type 1 font like Helvetica
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # Use DejaVu Sans, which is Type 1 compatible
    'pdf.fonttype': 42,  # Use Type 1 (TrueType) fonts
    'ps.fonttype': 42,   # Use Type 1 (TrueType) fonts in PostScript
})

# 数据：Proxy-GS
proxy_depth = [1.07, 1.19, 1.05, 1.08, 1.08]
proxy_filter = [2.60, 2.09, 2.15, 2.21, 2.05]
proxy_rendering = [5.35, 4.04, 3.51, 5.18, 3.61]

# 数据：Octree-GS
octree_depth = [0, 0, 0, 0, 0]
octree_filter = [4.65, 4.21, 4.45, 3.90, 3.62]
octree_rendering = [24.09, 22.00, 24.03, 21.15, 17.20]

blocks = ["Block 1", "Block 2", "Block 3", "Block 4", "Block 5"]
x = np.arange(len(blocks))  # 横坐标位置
bar_width = 0.25*0.66  # 柱子细一点

# 配色（论文风格）
colors = {
    "depth": "#4C78A8",      # 蓝色
    "filter": "#72B7B2",     # 青色
    "rendering": "#E45756"   # 红色
}

fig, ax = plt.subplots(figsize=(8, 5))

# --- Proxy-GS (实线边框, alpha高) ---
bottomA = np.zeros(len(x))
for vals, key in zip([proxy_depth, proxy_filter, proxy_rendering], ["depth", "filter", "rendering"]):
    ax.bar(x - bar_width/2, vals, bar_width, bottom=bottomA,
           color=colors[key], alpha=0.45,
           edgecolor="black", linewidth=1.2)
    bottomA += np.array(vals)

# --- Octree-GS (虚线边框, alpha低) ---
bottomB = np.zeros(len(x))
for vals, key in zip([octree_depth, octree_filter, octree_rendering], ["depth", "filter", "rendering"]):
    ax.bar(x + bar_width/2, vals, bar_width, bottom=bottomB,
           color=colors[key], alpha=0.45,
           edgecolor="black", linewidth=1.2, linestyle="--")
    bottomB += np.array(vals)

# 坐标轴
ax.set_xticks(x)
ax.set_xticklabels(blocks, fontsize=18)
ax.set_ylabel("Time (ms)", fontsize=18)
ax.set_xlabel("Datasets", fontsize=18)

# 图例
handles = [
    plt.Rectangle((0,0),1,1, color=colors["rendering"], alpha=0.45, label="Rendering"),
    plt.Rectangle((0,0),1,1, color=colors["filter"], alpha=0.45, label="Anchor filter"),
    plt.Rectangle((0,0),1,1, color=colors["depth"], alpha=0.45, label="Depth rendering"),
    Line2D([0], [0], color="black", lw=1.2, label="Proxy-GS"),           # 实线
    Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="Octree-GS")  # 虚线
]
ax.legend(handles=handles, loc="upper right", fontsize=10, frameon=False, ncol=2)
# Set y-axis limit to 40 to add extra space at the top for the legend
# ax.set_ylim(0, 40)

# Adjust layout to avoid clipping of the legend
plt.tight_layout()

# Save the figure with high resolution
plt.savefig("comparison_pretty.pdf", dpi=400, bbox_inches="tight")
plt.close(fig)
