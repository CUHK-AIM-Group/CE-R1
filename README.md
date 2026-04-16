<!-- <img src="./imgs/logo.png" alt="" width="130" > -->


<h1 align="center" style="font-size:26px; font-weight:bold; text-align:justify;">
  <img src="./imgs/logo.png" alt="" width="80" > An Adaptive Foundation Model with Evidence-based Clinical Reasoning for Gastroenterology
</h1>

Wenting Chen<sup>1,\*</sup> Shengyuan Liu<sup>2,\*</sup> Boyun Zheng<sup>2</sup> Jipeng Zhang<sup>3</sup> Wenxuan Wang<sup>3</sup> Dejun Fan<sup>4</sup> Raymond Shing Yan Tang<sup>5</sup> Yuen Tung Lam<sup>6</sup> Shannon Melissa Chan<sup>7</sup> Lei Xing<sup>1</sup> Jiancong Hu<sup>4,†</sup> Yixuan Yuan<sup>2,†</sup> 

<sup>1</sup> Department of Radiation Oncology, Stanford University, CA, USA  
<sup>2</sup> Department of Electronic Engineering, The Chinese University of Hong Kong, Hong Kong SAR, China  
<sup>3</sup> Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong SAR, China  
<sup>4</sup> The Sixth Affiliated Hospital, Sun Yat-sen University, Guangzhou, China  
<sup>5</sup> Department of Medicine and Therapeutics, The Chinese University of Hong Kong, Hong Kong SAR, China  
<sup>6</sup> The Nethersole School of Nursing, The Chinese University of Hong Kong, Hong Kong SAR, China  
<sup>7</sup> Department of Surgery, The Chinese University of Hong Kong, Hong Kong SAR, China  
<sup>\*</sup> These authors contributed equally.  
<sup>†</sup> Correspondence to Yixuan Yuan and Jiancong Hu.


<p align="center">
  <img src="./imgs/method_evaluation.png" alt="" width="85%" >
</p>


## 📄 Introduction
Gastrointestinal diseases affect 2.86 billion people globally, with capsule endoscopy (CE) providing crucial diagnostics but requiring manual review of over 60,000 frames per examination, a process associated with 17.4% disease miss rates. While artificial intelligence shows promise for CE analysis, existing endoscopic vision-language models (VLMs) lack multi-video understanding capability and cannot replicate the systematic multi-evidence reasoning that gastroenterologists integrate findings across anatomical regions to synthesize cohesive diagnoses.
Here we introduce CE-R1, an adaptive foundation model with evidence-based clinical reasoning capabilities specifically designed for gastroenterology. CE-R1 incorporates a dynamic router that assesses query complexity and selectively routes cases to either a lightweight model for straightforward questions or a deep reasoning model that generates transparent, step-by-step diagnostic thought processes. To enable this capability, we construct CE-Bench, the first large-scale multimodal CE dataset comprising 502,066 visual question-answering pairs with chain-of-thought reasoning annotations, spanning 70 fine-grained clinical sub-tasks across five core diagnostic categories: anatomy identification, endoscopic findings recognition, disease diagnosis, treatment planning, and medical report generation.
Comprehensive evaluation on both in-distribution and out-of-distribution datasets from four independent hospitals demonstrates that CE-R1 achieves 86.7% overall accuracy, substantially outperforming state-of-the-art VLMs (best baseline: 24.6%) and surpassing average physician performance (39.9%) by 21.1\%. CE-R1 maintains superior generalization across external validation sets (65.1–81.9\% accuracy). Critically, the multi-evidence clinical reasoning capability delivers substantial performance gains in complex diagnostic tasks: CE-R1 surpasses the model without reasoning by 8.5% in disease diagnosis, demonstrating the clinical value of transparent, step-by-step diagnostic processes. These results establish CE-R1 as a robust foundation model for comprehensive CE analysis with immediate applications in clinical decision support and medical education.

## ⚙️ Setup

### Environment Setup

Install the requirements:

```bash
conda env create -f environment.yml
pip install git+https://github.com/huggingface/transformers.git@v4.49.0
pip install -e .
pip install -e ".[torch,metrics]"
```

## Download Dataset and Models
Please download the public datasets and our pre-trained models from this repository (https://huggingface.co/datasets/Valentina007/CE_R1_data/).

Please make sure this folder (```CE_R1_data```) is under the same directory of current folder (```CE_R1```)
```bash
hf download Valentina007/CE_R1_data
```

Directory structure of this folder (```CE_R1_data```).

./anno and ./data include the part of data in CE-Bench. These folders include the public datasets, including kid-v1, kid-v2, and kvasir-capsule datasets.

./models includes the pre-trained models of CE-R1.

├── anno <br>
│   ├── kid-v1-image_test.json <br>
│   ├── kid-v2-image_test.json <br>
│   ├── kvasir-capsule-image_test.json <br>
│   └── kvasir-capsule-videoclip_test.json <br>
├── data <br>
│   ├── kid-dataset-1 <br>
│   ├── kid-dataset-2 <br>
│   ├── kvasir-capsule-labelled_images <br>
│   └── video_clips_v1 <br>
└── models <br>
    ├── deep <br>
    ├── lite <br>
    └── router_models <br>

## 🚀 Inference
```bash

INPUT_PATH_IMG="/path/to/input_image.png"
QUESTION_IMG="You question can be put here."

python test_single.py --path "$INPUT_PATH_IMG" --question "$QUESTION_IMG"

```
All the results will be saved at: ./results/model_output   

In ```./results/model_output/final_results.json```, you can get the output as follows:
```bash
{
  "input_path": "/path/to/input_image.png",
  "question": "You question",
  "probability": 0.3223,
  "model_version": "lite",
  "model_type": "lite",
  "media_type": "image",
  "generated_response": "Final output from CE-R1"
}
```
This result mentions the input of the CE-R1, the probability from router, model_type we used, and output of the CE-R1. 
When the probability from router is larger than 0.5, we use the CE-R1-Deep. Otherwise, we use CE-R1-Lite.

## Quick Start
Here, we provide an example about the WCE image or video as input.
```bash
sh ./lanuch/test_img_single.sh
```



## 🎈 Acknowledgements
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

[Multimodal-BERT-in-Medical-Image-and-Text-Classification](https://github.com/AxelAllen/Multimodal-BERT-in-Medical-Image-and-Text-Classification)

## 📮 Contact
Please contact me if you have any question (wentchen AT stanford dot edu)