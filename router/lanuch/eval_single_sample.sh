# export http_proxy="http://star-proxy.oa.com:3128"
# export https_proxy="http://star-proxy.oa.com:3128"

source /home/Vicky/miniconda3/etc/profile.d/conda.sh && conda activate ce_r1

TEXT_QUERY="<image>\nWhich abnormality is present in this WCE image?"
IMAGE_LIST="./sample_data/bleeding3.png"

# cd /NAS_REMOTE/vicky/wt/codes/github/CE_R1_data/models/router/clean
python3 test_mmbt_single.py \
  --text_query "${TEXT_QUERY}" \
  --image_list "${IMAGE_LIST}"
