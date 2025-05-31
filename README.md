# MVLIP: MambaVision as an Image Encoder in CLIP

A novel vision-language model that replaces CLIP's ViT image encoder with MambaVision for improved computational efficiency and performance.

## Overview
- 33-41% throughput improvement over baseline CLIP models
- 9-10% accuracy gains across all evaluation metrics vs. CLIP(Swin)
- Hybrid Mamba-Transformer architecture addressing ViT's quadratic complexity

## Quick Start
```
git clone https://github.com/Tosaaa/MVLIP.git
cd MVLIP
pip install -r requirements.txt
python finetune.py
```