#!/bin/bash
colmap image_undistorter \
    --image_path data/nyc/images \
    --input_path data/nyc/sparse/0 \
    --output_path data/nyc/dense \
    --output_type COLMAP \
    --max_image_size 3000



colmap patch_match_stereo \
    --workspace_path data/nyc/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true



colmap stereo_fusion \
    --workspace_path data/nyc/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path data/nyc/dense/fused.ply
