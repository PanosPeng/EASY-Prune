# Table of Contents
<!-- - [1. Preparation](#1-preparation)
- [2. Expert Selection](#2-expert-selection)
- [3. Model Pruning](#3-model-pruning)
- [4. Evaluation](#4-evaluation)
- [5. Next Steps](#5-next-steps)
- [6. Citation](#6-citation) -->

## 1. Preparation

### Requirements
```bash
cd EasyPrune
conda create -n easy-prune python=3.10
conda activate easy-prune
pip install -r requirements.txt
```
### Model Preparation

#### System Requirements
> [!NOTE] 
> Linux with Python 3.10 only. Mac and Windows are not supported.

Dependencies:
```pip-requirements
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

#### Model Weights Conversion
Download the model weights from Hugging Face, and put them into /path/to/DeepSeek-R1 folder.
Convert Hugging Face model weights to a specific format:

```shell
python convert.py --hf-ckpt-path /path/to/DeepSeek-V3 --save-path /path/to/DeepSeek-V3-Demo --n-experts 256 --model-parallel 16
```

### Data Preparation


## 2. Expert Selection


## 3. Model Pruning


## 4. Evaluation

## 5. Next Steps

## 6. Citation