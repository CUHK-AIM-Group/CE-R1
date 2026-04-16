export PATH="/home/Vicky/miniconda3/envs/ce_r1/bin:$PATH"

# lite 

## image
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA/qwen2_vl_lora_sft_kid-v1-image_test.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA/qwen2_vl_lora_sft_kid-v2-image_test.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA/qwen2_vl_lora_sft_kvasir-capsule-image_test.yaml
## video
# CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/inference/WCE_NEW_DATA/qwen2_vl_lora_sft_kvasir-capsule-videoclip_test.yaml


# deep

## image
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA_REASON/qwen2_vl_lora_sft_kid-v1-image_test.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA_REASON/qwen2_vl_lora_sft_kid-v2-image_test.yaml
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA_REASON/qwen2_vl_lora_sft_kvasir-capsule-image_test.yaml
## video
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/inference/WCE_NEW_DATA_REASON/qwen2_vl_lora_sft_kvasir-capsule-videoclip_test.yaml
