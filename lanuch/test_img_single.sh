# export PATH="/home/Vicky/miniconda3/envs/ce_r1/bin:$PATH"

# Set CUDA device(s), e.g., "0" or "0,1"
export CUDA_VISIBLE_DEVICES="2"

# =================================================================
# Test Image Input
# =================================================================
# INPUT_PATH_IMG="/NAS_REMOTE/vicky/wt/codes/github/CE_R1_data/data/kid-dataset-1/active bleeding/bleeding3.png"
# QUESTION_IMG="Which abnormality is present in this WCE image?"

# INPUT_PATH_IMG="./test_data/kvasir-capsule-labelled_images/5bb1d3cc7dc64cec_34112.jpg"
# QUESTION_IMG="What anatomical landmark is highlighted in this WCE image?"

INPUT_PATH_IMG="./test_data/kid-dataset-1/bleeding3.png"
QUESTION_IMG="Which abnormality is present in this WCE image?"

# Run the python script for Image
python test_single.py --path "$INPUT_PATH_IMG" --question "$QUESTION_IMG"

echo "----------------------------------------------------------------"

# =================================================================
# Test Video Input
# =================================================================
# INPUT_PATH_VID="/NAS_REMOTE/vicky/wt/codes/github/CE_R1_data/data/video_clips_v1/kvasir-capsule-videoclip/video_4874.mp4"
# QUESTION_VID="Based on this WCE video, what type of abnormal finding can be identified?"

INPUT_PATH_VID="./test_data/kvasir-capsule-videoclip/video_1192.mp4"
QUESTION_VID="Can you identify the anatomical landmark in this WCE video?"

# Run the python script for Video
python test_single.py --path "$INPUT_PATH_VID" --question "$QUESTION_VID"

