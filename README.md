<div align="center">
<h1> [CVPR 2026 Oral] Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting
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

### Dataset Structure

The example below shows the expected structure of the MatrixCity small-city dataset:

### Downloading MatrixCity Small City Dataset

First, download all data under `small_city/` from MatrixCity via Hugging Face:

```bash
pip install huggingface_hub

# Download the 'MatrixCity' dataset (will include all data)
# To only download the 'small_city' portion, use '--pattern' to filter
huggingface-cli download BoDai/MatrixCity --repo-type dataset --local-dir MatrixCity --include "small_city/**"
```

> Since the street data in the small city set is not originally split into blocks, running all images together may result in an excessively large dataset at once.  
> To address this, I partitioned the `small_city_road_horizon` subset into multiple blocks.  
> The corresponding `train` and `test` JSON files for each split are provided under `pose_block/`.  
> Each block (e.g., `block_1`, `block_2`, ...) contains its own `transforms_train.json` and `transforms_test.json`, making it easier to train and evaluate on manageable subsets of the data.  



To use the dataset, you need to combine the  data from Hugging Face's MatrixCity repository with the block-specific JSON splits we provide.

#### Move the provided `pose_block` directory into the correct place within the dataset:
```bash
mv pose_block MatrixCity/small_city/street/
```

The overall data directory structure should look like:


```text
MatrixCity/
└── small_city/
    ├── aerial/
    └── street/
        ├── pose_block/
        │   ├── block_1/
        │   │   ├── transforms_test.json
        │   │   └── transforms_train.json
        │   ├── block_2/
        │   │   ├── transforms_test.json
        │   │   └── transforms_train.json
        │   ├── block_3/
        │   │   ├── transforms_test.json
        │   │   └── transforms_train.json
        │   ├── block_4/
        │   │   ├── transforms_test.json
        │   │   └── transforms_train.json
        │   ├── block_5/
        │   │   ├── transforms_test.json
        │   │   └── transforms_train.json
        │   
        ├── train/
        │   ├── small_city_road_down/
        │   ├── small_city_road_horizon/
        │   ├── small_city_road_outside/
        │   └── small_city_road_vertical/
        ├── test/
        └── train_dense/
```

The following example uses the MatrixCity `block_5` scene. For reproducibility, we recommend using a dedicated output directory such as `output/block_5`. 

```bash
SCENE=proxy-gs/MatrixCity/small_city/street/pose_block/block_5
IMAGES=proxy-gs/MatrixCity/small_city/street/train/small_city_road_horizon
MESH=mesh/tsdf_fusion_post_block5.ply
POINTS=MatrixCity/small_city/aerial/small_city_pointcloud/point_cloud_ds20/aerial/Block_E.ply
DEPTH_DIR=mesh_depth_block_5
OUTPUT=output/block_5
```
In addition to obtaining the mesh using your own preferred method, you may also directly use the processed meshes we have released here:[Hugging Face dataset](https://huggingface.co/datasets/yy456/Proxy-GS/tree/main).

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



#### 4.  build `ProxyGS-Vulkan-Cuda-Interop`

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

## Citation

If you find this repository useful, please consider citing:

```bibtex
@article{gao2025proxy,
  title={Proxy-GS: Unified Occlusion Priors for Training and Inference in Structured 3D Gaussian Splatting},
  author={Gao, Yuanyuan and Gong, Yuning and Liu, Yifei and Jingfeng, Li and Zhang, Dingwen and Zhang, Yanci and Xu, Dan and Sun, Xiao and Zhong, Zhihang},
  journal={arXiv preprint arXiv:2509.24421},
  year={2025}
}
```
