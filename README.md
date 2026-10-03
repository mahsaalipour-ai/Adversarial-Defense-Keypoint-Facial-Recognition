# Adversarial-Defense-Keypoint-Facial-Recognition
Official Implementation of Data-Free Adversarial Defense against Deep Keypoint-based Attacks in Facial Recognition Models.
# Data-Free Adversarial Defense for Deep Keypoint-Based Attacks in Facial Recognition

Official PyTorch implementation of the M.Sc. thesis research: **Data-Free Adversarial Defense Mechanisms for Deep Keypoint-Based Facial Recognition Systems via Critical Region Selection, Feature Squeezing, and Adversarial Training.**

---

## Abstract
Deep facial recognition models are highly vulnerable to adversarial attacks, especially deep keypoint-based perturbations (e.g., DKA2). This repository provides a robust, data-free defense framework designed to counter keypoint attacks by integrating critical region selection, feature squeezing mechanisms, and robust adversarial training pipelines.

---

## Key Features & Methodology
- **Data-Free Defense Framework:** Protection against spatial perturbations without relying on full source training sets.
- **Critical Region Selection:** Focuses defense on critical facial landmark regions vulnerable to keypoint attacks.
- **Feature Squeezing:** Reduces adversarial search space while preserving facial identity feature embeddings.
- **Three-Solution Evaluation:** Comprehensive performance comparison across three defense methodologies.

---

## Repository Structure
├── models/                      # Pre-trained facial recognition architectures & defense networks
├── steps/                       # Step-by-step experiment implementation pipelines
├── three_solutions/             # Implementations of the three proposed defense mechanisms
├── tools/                       # Utility scripts, feature squeezing tools & metrics
├── weights/                     # Model weights & checkpoint parameters
├── docs/                        # Research documentation & thesis reference materials
└── README.md

## Contact & Citation
**Author:** Mahsa Alipour  
**Degree:** M.Sc. in Computer Science  
**Connect:**LinkedIn Profile:https://linkedin.com/in/mahsaalipour-ai
