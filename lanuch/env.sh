
conda create --name ce_r1 --clone qwen_vl


pip install git+https://github.com/huggingface/transformers.git@v4.49.0
pip install -e .

pip install -e ".[torch,metrics]"
pip install accelerate==0.34.0
