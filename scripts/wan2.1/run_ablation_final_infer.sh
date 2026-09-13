#!/bin/bash
export RAW_DATA_DIR=/mnt/cpfs/jiachengliu/code/object_removal/MaskBench/releases/bench300/
OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/c2r5 LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/c2r5/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_CORRUPTION_FRAMES=2 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/c4r3 LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/c4r3/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_RESTORATION_FRAMES=3 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/c4r7 LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/c4r7/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_RESTORATION_FRAMES=7 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/c6r5 LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/c6r5/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_CORRUPTION_FRAMES=6 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/no_cor LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/no_cor/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_CORRUPTION_FRAMES=0 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/no_restor LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/no_restor/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_RESTORATION_FRAMES=0 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/endpoint LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/endpoint/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_ENDPOINT_MODE=1 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/gamma1 LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/gamma1/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_INTERPOLATION_GAMMA=1 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/gamma1_5 LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/gamma1_5/checkpoint-1800/lora_diffusion_pytorch_model.safetensors SINGLETURN_CACHE_INTERPOLATION_GAMMA=1.5 bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/linear LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/linear/checkpoint-1800/lora_diffusion_pytorch_model.safetensors  bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/reverse LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/reverse/checkpoint-1800/lora_diffusion_pytorch_model.safetensors  bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/shuffle LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/shuffle/checkpoint-1800/lora_diffusion_pytorch_model.safetensors  bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh

OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/slerp LORA_PATH=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/slerp/checkpoint-1800/lora_diffusion_pytorch_model.safetensors  bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/wan2.1/infer_singleturn_1.3b.sh


# EVAL
for name in c2r5 c4r3 c4r7 c6r5 no_cor no_restor endpoint gamma1 gamma1_5 linear reverse shuffle slerp; do
  OUTPUT_DIR=/mnt/cpfs/jiachengliu/dataset/CORNE/ckpt/1.3B_ablation_final/outputs/${name} RAW_DATA_DIR=/mnt/cpfs/jiachengliu/code/object_removal/MaskBench/releases/bench300/ GPUS="0 1 2 3 4 5 6" RUN_NAME=eval_${name} MASK_TYPE=object_mask PRED_SUFFIX=_frame8_pure.png bash /mnt/cpfs/jiachengliu/code/object_removal/VideoCoF/WanEdit/scripts/eval_removal.sh
done
