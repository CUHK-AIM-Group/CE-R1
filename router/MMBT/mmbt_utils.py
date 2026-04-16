"""
The classes and functions in this module are adapted from Huggingface implementation: utils_mmimdb.py, which can be
found here: https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/utils_mmimdb.py

The ImageEncoderDenseNet class is modified from the original ImageEncoder class to be based on pre-trained DenseNet
instead of ResNet and to be albe to load saved pre-trained weights.

The forward function is also modified according to the forward function of the DenseNet model liste here:

Original forward function of DenseNet

def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

"""
import ast

import math
import json
import os
from collections import Counter
import logging

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
logger = logging.getLogger(__name__)
import numpy as np

import cv2  # import opencv
# directories and data filenames
MMBT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MMBT_DIR_PARENT, "data")
JSONL_DATA_DIR = os.path.join(DATA_DIR, "json")
IMG_DATA_DIR = os.path.join(DATA_DIR, "NLMCXR_png_frontal")


class JsonlDataset(Dataset):
    def __init__(self, jsonl_data_path, img_dir, tokenizer, transforms, labels, max_seq_length):
        self.data = [json.loads(line) for line in open(jsonl_data_path)]
        # self.data_dir = os.path.dirname(data_path)
        self.img_data_dir = img_dir
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length

        # for image normalization for DenseNet
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        if self.n_classes > 2:
            # multiclass
            label = torch.zeros(self.n_classes)
            label[self.labels.index(self.data[index]["label"])] = 1
        else:
            label = torch.LongTensor([self.labels.index(self.data[index]["label"])])

        image = Image.open(os.path.join(self.img_data_dir, self.data[index]["img"])).convert("RGB")
        image = self.transforms(image)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs


class PandaDataset(Dataset):
    def __init__(self, final_df, img_dir, tokenizer, transforms, labels, max_seq_length, use_balance=False, use_video=False, use_label=False):
        self.data = final_df #[json.loads(line) for line in open(jsonl_data_path)]
        # self.data_dir = os.path.dirname(data_path)
        self.img_data_dir = img_dir
        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = len(labels)
        self.max_seq_length = max_seq_length
        self.th = 0.75
        self.use_video = use_video
        if use_balance :
            ### extract the difficult data
            df_diff = final_df.loc[final_df["prob"] < self.th,:]
            df_easy = final_df.loc[final_df["prob"] >= self.th,:]
            df_easy_rand = df_easy.sample(n=len(df_diff), random_state=42, replace=False)
            print("number of difficult samples:",len(df_diff))
            print("number of easy samples:",len(df_easy))

            self.data = pd.concat([df_diff, df_easy_rand], ignore_index=True)
            print("total - self.data:",len(self.data))
        # for image normalization for DenseNet
        self.transforms = transforms
        self.use_label = False

        if self.use_label and use_balance:
            for index, row in final_df.iterrows():
                label = 0
                for label_content in self.label_list:
                    if label_content in row["conversations"]:
                        label = 1
                        break
                row["regular_label"] = label
            
            df_diff = final_df.loc[final_df["regular_label"] == 1,:]
            df_easy = final_df.loc[final_df["regular_label"] == 0,:]
            df_easy_rand = df_easy.sample(n=len(df_diff), random_state=42, replace=False)
            print("number of difficult samples:",len(df_diff))
            print("number of easy samples:",len(df_easy))
            self.data = pd.concat([df_diff, df_easy_rand], ignore_index=True)


        self.label_list = [ "Identify the digestive organ in this WCE image.",
                            "Which part of the digestive system is depicted in this WCE image?",
                            "What is the name of the digestive organ shown in this WCE image?"]

        #### part of data
        # debug
        # self.data = self.data.loc[:2176]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        """
        ,Unnamed: 0,id,images,width,height,conversations,label,class,reason,tmp_0.6,tmp_0.7,tmp_0.8,tmp_0.95,prob
        0,0,0,['/apdcephfs_qy3/share_301812049/jarviswang/wt/dataset/endoscopy/EndoGPT/kid-dataset-1/active bleeding/bleeding3.png'],360,360,"[{'from': 'human', 'value': '<image>\nWhich abnormality is present in this WCE image?'}, {'from': 'gpt', 'value': 'This WCE image shows abnormal endoscopic findings. The abnormality is the active bleeding.'}]",active bleeding,"['Bleeding and Blood-related', 'Vascular and Bleeding Disorders']",1,1,1,1,1,1.0
        """
        # print("self.data:",len(self.data))
        prob = self.data.loc[index, "prob"]
        keylist = self.data.keys().tolist()
        
        
        label = 1 if prob < self.th else 0

        item = self.data.loc[index]
        img_path_list = self.data.loc[index, "images"]
        isnull_dict = item.isnull()
        is_nan = isnull_dict["images"]
        # print("is null:",is_nan)
        # print("img_path_list:",type(img_path_list),img_path_list)
        if is_nan:
            img_path_list = self.data.loc[index, "videos"]
            # print("img_path_list:",img_path_list)
            img_path_list = img_path_list.split(',')
            img_path =  img_path_list[0]
            img_path = img_path.replace('"','').replace('[','').replace(']','').replace("'","").strip()
        else:
            img_path_list = img_path_list.replace("'", '"').strip()
            img_path_list = json.loads(img_path_list)
            img_path = img_path_list[0]
        # print("img_path:",img_path)
        is_video = 'mp4' in img_path
        if self.use_video and is_video: # is video

            # use opencv open the video 1.mp4
            # print("img_path:",os.path.exists(img_path),img_path)
            videoCapture = cv2.VideoCapture(img_path)#, cv2.CAP_DSHOW)
            # get the inf of vedio,fps and size
            fps = 5
            # read one frame from the video
            success, frame = videoCapture.read()
            # print("frame:",frame.shape,type(frame))
            # while success:
            #     # cv2.imshow("Oto Video", frame)  # display this frame
            #     # cv2.waitKey(int(fps))  # delay
            #     success, frame = videoCapture.read()  # get the next frame of the video
            #     break
            # some process after finish all the program
            # cv2.destroyAllWindows()     # close all the widows opened inside the program
            videoCapture.release        # release the video read/write handler
            image = frame
            image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

        text = self.data.loc[index, "conversations"]

        if self.use_label:
            label = row["regular_label"]
        
        if not is_video:
            text = text.replace("'", '"')

            # Convert to a Python list
            converted_list = json.loads(text)
            text = converted_list[0]["value"]
        else:
            text = text[text.find("'value': '")+10:text.find("}, {'from': 'gpt'")].replace('"','').replace("'","")

        sentence = torch.LongTensor(self.tokenizer.encode(text, add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        if self.n_classes > 2:
            # multiclass
            label = torch.zeros(self.n_classes)
            label[self.labels.index(label)] = 1
        else:
            label = torch.LongTensor([self.labels.index(label)])

        if not is_video:
            image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        # print("image:",image.shape)
        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": image,
            "label": label,
            "index": torch.LongTensor([index])
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs


def collate_fn(batch):
    """
    Specify batching for the torch Dataloader function

    :param batch: each batch of the JsonlDataset
    :return: text tensor, attention mask tensor, img tensor, modal start token, modal end token, label
    """
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    # infer_output_tensor = torch.stack([row["infer_output_list"] for row in batch])

    index_tensor = torch.stack([row["index"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor, index_tensor


def collate_fn_mask_all_text(batch):
    """
    Specify batching for the torch Dataloader function

    :param batch: each batch of the JsonlDataset
    :return: text tensor, attention mask tensor, img tensor, modal start token, modal end token, label
    """
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        #mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor


def get_multiclass_labels():
    """
    0: normal
    1: abnormal
    2: notable findings/abnormalities that are not relevant or within normal limits (WNL)

    :return:
    """
    return [0, 1, 2]


def get_labels():
    """
    0: normal
    1: abnormal

    :return: label classes
    """

    return [0, 1]


def get_image_transforms():
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )


def load_examples(tokenizer, wandb_config, evaluate=False, test=False, data_dir=JSONL_DATA_DIR, img_dir=IMG_DATA_DIR):
    """

    :param tokenizer: BERT tokenizer of choice
    :param wandb_config: wandb.config, which needs to contain file names of validation, test, and train files
    :param evaluate: True if loading Dataset for evaluating on validation or test set, False for Training
    :param test: True ONLY if loading Test Dataset, False if evaluating on validation set; if evaluate = False, test has to be False
    :param data_dir: Path to jsonl data directory e.g. "data/json"
    :param img_dir: Path to image directory e.g. "NLMCXR_png_frontal"
    :return: JasonlDataset derived from Torch Dataset class
    """
    if evaluate and not test:
        path = wandb_config.val_file #os.path.join(data_dir, wandb_config.val_file)
    elif evaluate and test:
        path = wandb_config.test_file #os.path.join(data_dir, wandb_config.test_file)
    elif not evaluate and not test:
        path = wandb_config.train_file #os.path.join(data_dir, wandb_config.train_file)
    else:
        # shouldn't get here not evaluate and test?
        raise ValueError("invalid data file option!!")

    img_transforms = get_image_transforms()

    if wandb_config.multiclass:
        labels = get_multiclass_labels()
    else:
        labels = get_labels()

    # get image list
    file_list = os.listdir(path)
    df_list = []
    for file_name in file_list:
        if not wandb_config.use_video and 'video' in file_name:
            continue
        print(file_name)
        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path)
        df["data_src"] = file_name
        df_list.append(df)
    
    final_df = pd.concat(df_list, ignore_index=True)
    # print(final_df)

    dataset = PandaDataset(final_df, img_dir, tokenizer, img_transforms, labels, wandb_config.max_seq_length -
                           wandb_config.num_image_embeds - 2, use_balance=wandb_config.use_balance, use_video=wandb_config.use_video, use_label=wandb_config.use_label)

    logger.info(f"PandaDataset from {path}\n")

    return dataset


def get_multiclass_criterion(jsonl_dataset_obj):
    label_freqs = jsonl_dataset_obj.get_label_frequencies()
    freqs = [label_freqs[label] for label in jsonl_dataset_obj.labels]
    label_weights = (torch.tensor(freqs, dtype=torch.float) / len(jsonl_dataset_obj)) ** -1
    return nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
