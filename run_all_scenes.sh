#!/bin/bash

# 通用训练参数
gpu=-1
ratio=1
resolution=2
appearance_dim=0

fork=2
base_layer=-1
visible_threshold=0.9
dist2level="round"
update_ratio=0.2

progressive="False"
dist_ratio=0.999
levels=-1
init_level=-1
extra_ratio=0.25
extra_up=0.01

# 场景列表（scene_path exp_name）
scenes=(
    "alameda alameda"
    # "london london"
    # "nyc nyc"
)

for item in "${scenes[@]}"; do
    set -- $item  # 分割为两个变量
    scene=$1
    exp_name=$2

    echo "===================="
    echo "开始训练场景: $scene"
    echo "实验名: $exp_name"
    echo "===================="

    ./train.sh -d ${scene} -l ${exp_name} --gpu ${gpu} -r ${resolution} --ratio ${ratio} --appearance_dim ${appearance_dim} \
        --fork ${fork} --visible_threshold ${visible_threshold} --base_layer ${base_layer} --dist2level ${dist2level} --update_ratio ${update_ratio} \
        --progressive ${progressive} --levels ${levels} --init_level ${init_level} --dist_ratio ${dist_ratio} \
        --extra_ratio ${extra_ratio} --extra_up ${extra_up}

    if [ $? -ne 0 ]; then
        echo "❌ 场景 $scene 训练失败，退出脚本"
        exit 1
    else
        echo "✅ 场景 $scene 训练完成"
    fi
done

echo "🎉 所有场景训练完毕"
