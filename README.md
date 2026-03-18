<div align="center">
<h1> [CVPR 2026] Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting
</h1>

<!-- <a href="https://www.arxiv.org/pdf/2509.24421" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT-blue" alt="Paper PDF"> -->
</a>
<a href="https://www.arxiv.org/pdf/2509.24421" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/arXiv-2509.24421-b31b1b" alt="arXiv">
</a>
<!-- alt="arXiv"></a> -->
<a href="https://visionary-laboratory.github.io/Proxy-GS/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


<!-- **Northwestern Polytechnical University**; **Shanghai Artificial Intelligence Laboratory** -->

<!-- | [ICCV 2025] CityGS-X : A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction -->

<!-- [Yuanyuan Gao*](https://scholar.google.com/citations?hl=en&user=1zDq0q8AAAAJ), [Hao Li*](https://lifuguan.github.io/), [Jiaqi Chen*](https://github.com/chenttt2001), [Zhengyu Zou](https://vision-intelligence.com.cn), [Zhihang Zhong†](https://zzh-tech.github.io), [Dingwen Zhang†](https://vision-intelligence.com.cn), [Xiao Sun](https://jimmysuen.github.io), [Junwei Han](https://vision-intelligence.com.cn)<br>(\* indicates equal contribution, † means co-corresponding author)<br> -->

</div>
 
![Teaser image](assets/teaser.jpg)

This repo contains official implementations of Proxy-GS, ⭐ us if you like it!

<!-- ## Project Updates
- 🔥🔥 News: ```2025/4/17```: training & inference code is now available! You can try it.
- 🔥🔥 News: ```2025/6/28```: CityGS-X has been accepted to ICCV 2025. -->
  
## Todo List
- 🔥🔥 News: ```2026/2/26```: Proxy-GS has been accepted to CVPR 2026.
- [✓] Release the training & inference code of Proxy-GS.
- [ ] Release all model checkpoints. 

## Installation

We recommend using a dedicated conda environment:

```bash
conda create -n proxy-gs python=3.10 -y
conda activate proxy-gs
```

Install a CUDA-enabled PyTorch build that matches your local CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt

# Install torch-scatter with the wheel matching your PyTorch/CUDA version.
# See: https://data.pyg.org/whl/
pip install torch-scatter

# Install local CUDA extensions
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Training and Inference 

The following example uses the MatrixCity `block_E` scene. For reproducibility, we recommend using a dedicated output directory such as `output/block_E`.

```bash
SCENE=proxy-gs/MatrixCity/small_city/street/pose/block_E
IMAGES=proxy-gs/MatrixCity/small_city/street/train/small_city_road_horizon
MESH=cvpr/block_E_from_mesh.ply
POINTS=proxy-gs/MatrixCity/small_city/street/pose/block_E/points3d.ply
DEPTH_DIR=mesh_depth_block_E
OUTPUT=output/block_E
```

### 1. Render mesh depth and save caches

Render the proxy mesh into per-view depth maps and save them as `.npy` files:

```bash
python mesh_render.py \
  -s ${SCENE} \
  -m ${OUTPUT} \
  -i ${IMAGES} \
  --ply_mesh ${MESH} \
  --depth_npy_dir ${DEPTH_DIR}
```

The rendered depth files will be written to `${DEPTH_DIR}`.

### 2. Train Proxy-GS

After the depth cache is ready, start training with the rendered mesh depth prior:

```bash
python train.py \
  -s ${SCENE} \
  -m ${OUTPUT} \
  -i ${IMAGES} \
  --ply_mesh ${MESH} \
  --depth_npy_dir ${DEPTH_DIR} \
  --ply_path ${POINTS}
```

Checkpoints will be saved under `${OUTPUT}/point_cloud`.

### 3. Test and evaluate



#### 4.  build `ProxyGS-Vulkan-Cuda-Interop` ()

**This optional Vulkan backend currently requires Ubuntu Linux and an NVIDIA RTX-series compute GPU.**

If you want to use the Vulkan-CUDA interop backend used by `render_real.py`, build the Python extension in `ProxyGS-Vulkan-Cuda-Interop` first.

```bash
cd ProxyGS-Vulkan-Cuda-Interop

# Use the current conda environment
export PYTHON_EXECUTABLE=$(which python)

# Set this to your local Vulkan SDK path
export VULKAN_SDK_PREFIX=$HOME/VulkanSDK/1.4.321.1/x86_64

# Build the Python extension only
./build_pyext_only.sh build-py
```

After building, verify that the extension can be imported successfully:

```bash
cd ProxyGS-Vulkan-Cuda-Interop
python -c "
import sys
sys.path.insert(0, 'build-py/_bin/Release')
import vk2torch_ext
print('vk2torch_ext OK:', vk2torch_ext.__version__)
"
```

### FPS Evaluation

Once the extension is built, go back to the project root and run `render_real.py`.

```bash
cd ..
python render_real.py \
  -m ${OUTPUT} \
  --scene_file /absolute/path/to/your_scene.glb
```

If you want to keep the exact commands used in our current internal runs, simply replace `${OUTPUT}` with `output`.

## Acknowledgements

This project builds upon several excellent open-source repositories. We sincerely thank the authors of:

- [Octree-GS](https://github.com/city-super/Octree-GS): *Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians*
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS): *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering*
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
- [vk_lod_clusters](https://github.com/nvpro-samples/vk_lod_clusters/tree/main): *Sample for cluster-based continuous level of detail rasterization or ray tracing*





<!-- ## BibTeX

```bibtex
@misc{gao2025citygsxscalablearchitectureefficient,
      title={CityGS-X: A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction}, 
      author={Yuanyuan Gao and Hao Li and Jiaqi Chen and Zhengyu Zou and Zhihang Zhong and Dingwen Zhang and Xiao Sun and Junwei Han},
      year={2025},
      eprint={2503.23044},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23044}, 
}
``` -->
