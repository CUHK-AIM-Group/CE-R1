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
Our work is based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [Multimodal-BERT-in-Medical-Image-and-Text-Classification](https://github.com/AxelAllen/Multimodal-BERT-in-Medical-Image-and-Text-Classification). We use LLaMA-Factory to train our CE-R1-Lite and CE-R1-Deep, and use Multimodal-BERT to train the dynamic router.

### Environment Setup

Install the requirements:
```bash
pip install -e ".[torch,metrics]"
```
<small>*Please refer to LLaMA-Factory and Multimodal-BERT for more environment details.</small>


## 🚀 Inference

### CE-R1-Lite
```
llamafactory-cli train examples/inference/WCE_NEW_DATA/qwen2_vl_lora_sft_kvasir-capsule-videoclip_test.yaml
```
### CE-R1-Deep
```
llamafactory-cli train examples/inference/WCE_NEW_DATA_REASON/qwen2_vl_lora_sft_kvasir-capsule-videoclip_test.yaml
```
### CE-R1 (w/ Router)
```
python3 router.py
```
## 🎈 Acknowledgements
Some source code of ours is borrowed from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Multimodal-BERT-in-Medical-Image-and-Text-Classification](https://github.com/AxelAllen/Multimodal-BERT-in-Medical-Image-and-Text-Classification). Thanks for their contributions.

## 📮 Contact
Please contact me if you have any question (wentchen AT stanford dot edu)