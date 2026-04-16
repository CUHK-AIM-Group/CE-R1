import argparse
import json
import os
import random
import subprocess

def create_tmp_json(is_video, path, question):
    tmp_dir = "tmp_data"
    os.makedirs(tmp_dir, exist_ok=True)
    
    if is_video:
        json_path = os.path.join(tmp_dir, "tmp_vid_data.json")
        # Adjust the format based on the required LlamaFactory sharegpt or default structure
        data = [{
            "videos": [path],
            "conversations": [
                {"from": "human", "value": f"<video>{question}"},
                {"from": "gpt", "value": ""}
            ]
        }]
    else:
        json_path = os.path.join(tmp_dir, "tmp_img_data.json")
        data = [{
            "images": [path],
            "conversations": [
                {"from": "human", "value": f"<image>{question}"},
                {"from": "gpt", "value": ""}
            ]
        }]
        
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Generated tmp json data at {json_path}")

import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# Add 'router' directory to Python path to import MMBT modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'router'))
from MMBT.image import ImageEncoderDenseNet
from MMBT.mmbt_config import MMBTConfig
from MMBT.mmbt import MMBTForClassification
from MMBT.mmbt_utils_single import load_examples, collate_fn, get_labels
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler

# Cached model & tokenizer to prevent reloading on multiple calls
_ROUTER_MODEL = None
_ROUTER_TOKENIZER = None

class MMBTArgs:
    def __init__(self, **kwargs):
        self.data_dir = "data/json"
        self.model_name = "bert-base-uncased"
        self.config_name = "bert-base-uncased"
        self.tokenizer_name = "bert-base-uncased"
        self.max_seq_length = 300
        self.num_image_embeds = 3
        self.eval_batch_size = 32
        self.use_label = False
        self.multiclass = False
        self.use_balance = False
        self.num_workers = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for k, v in kwargs.items():
            setattr(self, k, v)

def load_router_model():
    global _ROUTER_MODEL, _ROUTER_TOKENIZER
    if _ROUTER_MODEL is not None:
        return _ROUTER_MODEL, _ROUTER_TOKENIZER
        
    print("Loading MMBT Router model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    num_labels = len(get_labels())
    model_name = "bert-base-uncased"
    
    transformer_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
    img_encoder = ImageEncoderDenseNet(num_image_embeds=3)
    multimodal_config = MMBTConfig(transformer, img_encoder, num_labels=num_labels, modal_hidden_size=1024)
    model = MMBTForClassification(transformer_config, multimodal_config)
    
    checkpoint_dir = "../CE_R1_data/models/router_models"
    checkpoint = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
    model.eval()

    _ROUTER_MODEL = model
    _ROUTER_TOKENIZER = tokenizer
    return _ROUTER_MODEL, _ROUTER_TOKENIZER

def get_probability_from_router(image_or_video_path, question):
    is_video = image_or_video_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    sample = {
        "id": "single_sample_0",
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": ""
            }
        ],
        "prob": 1.0,
        "label": "dummy"
    }
    
    if is_video:
        sample["videos"] = [image_or_video_path]
    else:
        sample["images"] = [image_or_video_path]
        
    os.makedirs("./sample_data", exist_ok=True)
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.json', dir='./sample_data')
    with os.fdopen(tmp_fd, 'w') as f:
        json.dump([sample], f)
        
    args = MMBTArgs(test_file=tmp_path, use_video=is_video)
    
    model, tokenizer = load_router_model()
    eval_dataset = load_examples(tokenizer, args, evaluate=True, test=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn, num_workers=args.num_workers
    )

    print("Running MMBT Router Inference...")
    prob_val = 0.0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            input_ids = batch[0]
            attention_mask = batch[1]
            input_modal = batch[2]
            modal_start_tokens = batch[3]
            modal_end_tokens = batch[4]
            labels = batch[5]

            outputs = model(
                input_modal,
                input_ids=input_ids,
                modal_start_tokens=modal_start_tokens,
                modal_end_tokens=modal_end_tokens,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            logits = outputs.logits
            prob = torch.nn.functional.softmax(logits, dim=1)
            prob_val = float(prob[:,0].cpu().detach().numpy()[0])
            break # only 1 sample expected
            
    try:
        os.remove(tmp_path)
    except Exception as e:
        print(f"Cleanup error: {e}")

    return prob_val

def run_inference(image_or_video_path, question, prob=None):
    if prob is None:
        prob = get_probability_from_router(image_or_video_path, question)
    
    is_video = image_or_video_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    version = "deep" if prob >= 0.5 else "lite"
    model_type = "video" if is_video else "image"
    
    print(f"Input path: {image_or_video_path}")
    print(f"Question: {question}")
    print(f"Probability: {prob}")
    print(f"Selected Version: {version}")
    print(f"Selected Model Type: {model_type}")

    # Generate json
    create_tmp_json(is_video, image_or_video_path, question)
    
    if version == "lite":
        if is_video:
            yaml_path = "examples/inference/WCE_NEW_DATA/lite_video_test.yaml"
        else:
            yaml_path = "examples/inference/WCE_NEW_DATA/lite_img_test.yaml"
    else:
        if is_video:
            yaml_path = "examples/inference/WCE_NEW_DATA_REASON/reason_lite_video.yaml"
        else:
            yaml_path = "examples/inference/WCE_NEW_DATA_REASON/reason_lite_img.yaml"
            
    print(f"Suggested YAML config to run inference: {yaml_path}")
    
    cmd = f"llamafactory-cli train {yaml_path}"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True)
    
    # Read the output file from LlamaFactory to extract the model's generated text
    import yaml
    try:
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        output_dir = yaml_config.get('output_dir', './results/model_output')
        pred_file = os.path.join(output_dir, 'generated_predictions.jsonl')
        
        generated_text = ""
        if os.path.exists(pred_file):
            with open(pred_file, 'r', encoding='utf-8') as f:
                # Get the last record or all records
                lines = f.readlines()
                if lines:
                    last_pred = json.loads(lines[-1].strip())
                    generated_text = last_pred.get("predict", "")
        
        final_result = {
            "input_path": image_or_video_path,
            "question": question,
            "probability": prob,
            "model_version": version,
            "model_type": version, # "lite" or "deep" map to model_version originally
            "media_type": model_type, # map from "image" or "video"
            "generated_response": generated_text
        }
        
        final_output_path = os.path.join(output_dir, 'final_results.json')
        with open(final_output_path, 'a', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
            f.write('\n')
            
        print(f"Appended inference result to {final_output_path}")
    except Exception as e:
        print(f"Error extracting final output: {e}")

def get_samples(dataset_files):
    for name, path in dataset_files.items():
        data = load_dataset(path)
        if data:
            sample = random.choice(data)
            print(f"\n--- Sample from {name} ---")
            print(json.dumps(sample, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test single image/video with probability-based model selection")
    parser.add_argument("--path", type=str, help="Path to the image or video", required=False)
    parser.add_argument("--question", type=str, help="Question about the image/video", required=False)
    parser.add_argument("--prob", type=float, help="Probability score (>= 0.5 uses deep version)", required=False)
    parser.add_argument("--sample", action="store_true", help="Extract samples from datasets")
    
    args = parser.parse_args()
    
    if args.sample:
        # Load random samples from the requested datasets based on dataset_info.json paths
        dataset_files = {
            "kid-v1-image_test (lite)": "../../CE_R1_data/anno/kid-v1-image_test.json",
            "kid-v2-image_test (lite)": "../../CE_R1_data/anno/kid-v2-image_test.json",
            "kvasir-capsule-image_test (lite)": "../../CE_R1_data/anno/kvasir-capsule-image_test.json",
            "kvasir-capsule-videoclip_test (lite)": "../../CE_R1_data/anno/kvasir-capsule-videoclip_test.json",
            "kid-v1-image_test_reason (deep)": "/apdcephfs_qy3/share_301812049/jarviswang/wt/dataset/endoscopy/code/think_process/merge_json/kid-v1-image_test.json"
        }
        get_samples(dataset_files)
    elif args.path and args.question:
        run_inference(args.path, args.question, args.prob)
    else:
        parser.print_help()