import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

from textBert_utils import set_seed
from MMBT.image import ImageEncoderDenseNet
from MMBT.mmbt_config import MMBTConfig
from MMBT.mmbt import MMBTForClassification

from MMBT.mmbt_utils_single import JsonlDataset, get_image_transforms, get_labels, load_examples, collate_fn, get_multiclass_labels, get_multiclass_criterion

import argparse
import glob
import logging
import random
import json
import os
from collections import Counter
import numpy as np
from matplotlib.pyplot import imshow

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(f'Project Hyperparameters and Other Configurations Argument Parser')

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--data_dir",
    default="data/json",
    type=str,
    help="The input data dir. Should contain the .jsonl files.",
)
parser.add_argument(
    "--model_name",
    default="bert-base-uncased",
    type=str,
    help="model identifier from huggingface.co/models",
)
parser.add_argument(
    "--output_dir",
    default="mmbt_output_findings_10epochs_n",
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
)

    
parser.add_argument(
    "--config_name", default="bert-base-uncased", type=str, help="Pretrained config name if not the same as model_name"
)
parser.add_argument(
    "--tokenizer_name",
    default="bert-base-uncased",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)

parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
parser.add_argument(
    "--eval_batch_size", default=32, type=int, help="Batch size for evaluation."
)
parser.add_argument(
    "--max_seq_length",
    default=300,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--num_image_embeds", default=3, type=int, help="Number of Image Embeddings from the Image Encoder"
)
parser.add_argument("--do_train", default=False, type=bool, help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=bool, help="Whether to run eval on the dev set.")
parser.add_argument(
    "--evaluate_during_training", default=True, type=bool, help="Run evaluation during training at each logging step."
)

parser.add_argument(
    "--use_label", default=False, type=bool, help="use_label."
)

parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
)
parser.add_argument("--patience", default=5, type=int, help="Patience for Early Stopping.")
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=25, help="Log every X updates steps.")

parser.add_argument("--val_steps", type=int, default=25, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=25, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    default=True, type=bool,
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)

parser.add_argument("--num_workers", type=int, default=8, help="number of worker threads for dataloading")

parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--use_balance", default=False, type=bool, help="Whether to make the dataset balance.")
parser.add_argument("--use_video", default=False, type=bool, help="Whether to video for training.")
parser.add_argument("--test_file", default="/NAS_REMOTE/vicky/wt/codes/github/CE_R1_data/models/router/anno/overall_prob_0.5", type=str, help="Testing file or directory path.")


parser.add_argument("--text_query", type=str, default="", help="Text question for single sample testing")
parser.add_argument("--image_list", type=str, default="", help="Comma separated image paths for single sample testing")
parser.add_argument("--video_list", type=str, default="", help="Comma separated video paths for single sample testing")
args = parser.parse_args()

print("args:",args)

if args.text_query and (args.image_list or args.video_list):
    sample = {
        "id": "single_sample_0",
        "conversations": [
            {
                "from": "human",
                "value": args.text_query
            },
            {
                "from": "gpt",
                "value": ""
            }
        ],
        "prob": 1.0,
        "label": "dummy",
    }
    if args.image_list:
        sample["images"] = args.image_list.split(",")
    if args.video_list:
        sample["videos"] = args.video_list.split(",")
        args.use_video = True
    
    import tempfile
    os.makedirs("./sample_data", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.json', dir='./sample_data')
    with os.fdopen(tmp_fd, 'w') as f:
        json.dump([sample], f)
    
    args.test_file = tmp_path


# Setup CUDA, GPU & distributed training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
args.device = device

# for multiclass labeling
args.multiclass = False



### train set folder
train_set_dir = '/apdcephfs_qy3/share_301812049/jarviswang/wt/codes/endogpt/LLaMA-Factory/output/qwen_evaluation/overall_trainset_prob_0.5'
test_set_dir = args.test_file
#'/NAS_REMOTE/vicky/wt/codes/v2_10_01/output/qwen_evaluation/overall_prob_0.5'
#'/NAS_REMOTE/vicky/wt/codes/github/CE_R1_data/anno'
#'/apdcephfs_qy3/share_301812049/jarviswang/wt/codes/endogpt/LLaMA-Factory/output/qwen_evaluation/overall_prob_0.5'

# train_file_list = os.listdir(train_set_dir)
# test_file_list = os.listdir(test_set_dir)

# train_file = "image_labels_findings_frontal_train.jsonl"
# val_file = "image_labels_findings_frontal_val.jsonl"
# test_file = "image_labels_findings_frontal_test.jsonl"

# Setup Train/Val/Test filenames
args.train_file = train_set_dir #train_file
args.val_file = test_set_dir #val_file
if not getattr(args, 'text_query', False):
    args.test_file = test_set_dir #test_file



tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
        do_lower_case=True,
        cache_dir=None,
    )

def evaluate(args, model, tokenizer, evaluate=True, test=False, prefix="", epoch_idx="", step_idx=""):
    
    if test:
        # start a separate tensorboard to track testing eval result
        comment = f"test_{args.output_dir}_{args.eval_batch_size}"
        # tb_writer = SummaryWriter(comment=comment)

    eval_output_dir = args.output_dir
    eval_dataset = load_examples(tokenizer, args, evaluate=evaluate, test=test)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn, num_workers=args.num_workers
    )

    final_df = eval_dataset.data
    final_df["complex_pred"] = 0.0
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    probs_list = []
    # infer_output_lists = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[5]
            input_ids = batch[0]
            input_modal = batch[2]
            attention_mask = batch[1]
            modal_start_tokens = batch[3]
            modal_end_tokens = batch[4]
            
            if args.multiclass:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    return_dict=True
                )
            else:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=labels,
                    return_dict=True
                )
            #logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            #tmp_eval_loss = criterion(logits, labels)
            logits = outputs.logits
            if args.multiclass:
                criterion = get_multiclass_criterion(eval_dataset)
                tmp_eval_loss = criterion(logits, labels)
            else:
                tmp_eval_loss = outputs.loss
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # Move logits and labels to CPU
        if args.multiclass:
            pred = torch.sigmoid(logits).cpu().detach().numpy() > 0.5
        else:            
            pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy()
        out_label_id = labels.detach().cpu().numpy()
        preds.append(pred)
        out_label_ids.append(out_label_id)
        # infer_output_lists.append(batch[6].detach().cpu().numpy())

        # index list
        index_list = np.squeeze(batch[6].detach().cpu().numpy()).tolist()
        pred = pred.tolist()
        # print("--pred:",pred)
        # print("--index_list:",index_list)
        # print("--pred:",len(pred))
        # print("--index_list:",len(index_list))
        # print("pred:",type(pred),pred.shape)
        # print("index_list:",type(index_list), index_list.shape)
        prob = torch.nn.functional.softmax(logits, dim=1)
        prob = prob[:,0].cpu().detach().numpy()
        # print("prob",prob.shape)
        # exit(0)
        final_df.loc[index_list, "complex_pred"] = pred
        final_df.loc[index_list, "complex_prob"] = prob
        # exit(0)

        ### reason,tmp_0.6,tmp_0.7,tmp_0.8,tmp_0.95

    eval_loss = eval_loss / nb_eval_steps

    result = {"loss": eval_loss}

    if args.multiclass:
        tgts = np.vstack(out_label_ids)
        preds = np.vstack(preds)
        result["macro_f1"] = f1_score(tgts, preds, average="macro")
        result["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        preds = [l for sl in preds for l in sl]
        out_label_ids = [l for sl in out_label_ids for l in sl]
        result["accuracy"] = accuracy_score(out_label_ids, preds)

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("Epoch: %s, Step: %s,  %s = %s", epoch_idx, step_idx, key, str(result[key]))
            writer.write("Epoch: %s, Step: %s,%s = %s\n" % (epoch_idx, step_idx, key, str(result[key])))
            # if test:
            #     tb_writer.add_scalar(f'eval_{key}', result[key], nb_eval_steps)
    
    # if test:
    #     tb_writer.close()


    return result, final_df


# Setup logging
logger = logging.getLogger(__name__)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#                     datefmt="%m/%d/%Y %H:%M:%S",
#                     filename=os.path.join(args.output_dir, f"{os.path.splitext(args.train_file)[0]}_logging.txt"),
#                     level=logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    filename=os.path.join(args.output_dir, f"logging.txt"),
                    level=logging.INFO)
logger.warning("device: %s, n_gpu: %s",
        args.device,
        args.n_gpu
)
# Set the verbosity to info of the Transformers logger (on main process only):

# Set seed
set_seed(args)


# Setup model
if args.multiclass:
    labels = get_multiclass_labels()
    num_labels = len(labels)
else:
    labels = get_labels()
    num_labels = len(labels)
transformer_config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
        do_lower_case=True,
        cache_dir=None,
    )
transformer = AutoModel.from_pretrained(args.model_name, config=transformer_config, cache_dir=None)
img_encoder = ImageEncoderDenseNet(num_image_embeds=args.num_image_embeds)
multimodal_config = MMBTConfig(transformer, img_encoder, num_labels=num_labels, modal_hidden_size=1024)
model = MMBTForClassification(transformer_config, multimodal_config)

# model.to(args.device)

logger.info(f"Testing/evaluation parameters: {args}")

results = {}
if args.do_eval:
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(os.path.dirname(c) 
        for c in sorted(glob.glob(args.output_dir + "/**/" + 
                                  WEIGHTS_NAME, recursive=False)))
        # recursive=False because otherwise the parent diretory gets included
        # which is not what we want; only subdirectories

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    # for checkpoint in checkpoints:
    #     global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #     prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
    # checkpoint_dir = "/apdcephfs_qy3/share_301812049/jarviswang/wt/codes/endogpt/router/Multimodal-BERT-in-Medical-Image-and-Text-Classification/checkpoints/mmbt_epoch_2_balance/checkpoint-50"
    #### NOT BALANCE
    checkpoint_dir = "/NAS_REMOTE/vicky/wt/codes/github/CE_R1_data/models/router/checkpoints/mmbt_epoch_20_BS_768_use_label_use_balance/checkpoint-30" #"/apdcephfs_qy3/share_301812049/jarviswang/wt/codes/endogpt/router/Multimodal-BERT-in-Medical-Image-and-Text-Classification/checkpoints/mmbt_epoch_10_BS_768/checkpoint-20"
    #'/apdcephfs_qy3/share_301812049/jarviswang/wt/codes/endogpt/router/Multimodal-BERT-in-Medical-Image-and-Text-Classification/checkpoints/mmbt_epoch_2/checkpoint-325'
    model = MMBTForClassification(transformer_config, multimodal_config)
    checkpoint = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    model.load_state_dict(torch.load(checkpoint))
    model.to(args.device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(args.n_gpu)))

    result, final_df = evaluate(args, model, tokenizer, evaluate=True, test=True, prefix="")
    # result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
    # print(result)
    # results.update(result)
    # gt_list = np.squeeze(np.array(result["gt"]))
    # pred_list = np.array(result["pred"])
    # infer_output = np.concatenate(result["infer_output"], axis=0)
    # print("gt_list:",gt_list.shape,gt_list)
    # print("pred_list:",pred_list.shape, pred_list)
    # print("infer_output:",infer_output.shape, infer_output)
    ## diff 

    final_df.to_csv("./outputs/eval_mmbt.csv")
    # results_file = open('./results/eval.txt', 'w')
    # sample_num = gt_list.shape[0]
    # total_easy = []
    # total_easy_avg = []
    # total_diff = []
    # for i in range(sample_num):
    #     gt = gt_list[i]
    #     pred = pred_list[i]
    #     reason = infer_output[i,0]
    #     no_reason = infer_output[i,4]
    #     no_reason_avg = np.mean(infer_output[i,1:4])
    #     val = 0.0
    #     val_avg = 0.1
    #     if pred == 1:
    #         val = reason
    #     else:
    #         val = no_reason
    #         val_avg = no_reason_avg
    #     if gt == 1 :
    #         total_diff.append(val)
    #     else:
    #         total_easy.append(val)
    #         total_easy_avg.append(val_avg)
    # print("avg-diff:",sum(total_diff)*1.0/len(total_diff))
    # print("avg-easy:",sum(total_easy)*1.0/len(total_easy))
    # print("avg-easy_avg:",sum(total_easy_avg)*1.0/len(total_easy_avg))
    # results_file.write("avg-diff:{}\n".format(sum(total_diff)*1.0/len(total_diff)))
    # results_file.write("avg-easy:{}\n".format(sum(total_easy)*1.0/len(total_easy)))
    # results_file.write("avg-easy-avg:{}\n".format(sum(total_easy_avg)*1.0/len(total_easy_avg)))
    # results_file.write("accuracy:{}\n".format(result["accuracy"]))
    # results_file.close()


results.keys()
# Training
# if args.do_train:
    # train_dataset = load_examples(tokenizer, args)
    # # criterion = nn.CrossEntropyLoss
    # global_step, tr_loss = train(args, train_dataset, model, tokenizer)

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
#     logger.info("Saving model checkpoint to %s", args.output_dir)
#     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
#     # They can then be reloaded using `from_pretrained()`
#     model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
#     torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
#     tokenizer.save_pretrained(args.output_dir)
#     transformer_config.save_pretrained(args.output_dir)
#     # Good practice: save your training arguments together with the trained model
#     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

#     # Load a trained model and vocabulary that you have fine-tuned
#     model = MMBTForClassification(transformer_config, multimodal_config)
#     model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME)))
#     tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
#     model.to(args.device)
# logger.info("***** Training Finished *****")