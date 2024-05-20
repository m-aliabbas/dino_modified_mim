# DINO with Modified Image Reconstruction
This repository contains the implementation of DINO (Distillation with No Labels) with modifications to include an image reconstruction task, inspired by Multi-Concept Self-Supervised Learning (MC-SSL). 
The project leverages Vision Transformers (ViTs) to learn robust visual representations without labeled data and explores the integration of reconstruction tasks to enhance learning.
DINO is a self-supervised learning framework that utilizes Vision Transformers to learn high-quality image representations without labeled data.
This repository extends the DINO approach by incorporating an image reconstruction task to investigate the potential benefits of combining global feature learning with local detail reconstruction.
# Features
## Self-Supervised Learning: 
Leverages the DINO framework for learning robust representations without labels.
## Image Reconstruction Task: 
Integrates a reconstruction head using Group Masked Model Learning (GMML) methods to enhance learning.
## Teacher-Student Framework:
Employs a teacher network as a momentum-averaged version of the student network for stable learning.
## Vision Transformer (ViT-Tiny):
Uses ViT-Tiny for efficient training and experimentation.
# Requirements
- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- torchvision 0.9.0 or higher
- numpy
- matplotlib

# Training
```
python main_dino.py
```
# Evaluation
```
python eval_linear.py
```

DINO Paper https://arxiv.org/abs/2104.14294
MCSSL Paper https://arxiv.org/abs/2104.14294

## Credit and Motiviation
This project is part of my PhD admission task at the University of Surrey, aiming to explore and enhance self-supervised learning techniques for Vision Transformers. The goal is to investigate the integration of image reconstruction tasks with DINO to improve representation learning.
