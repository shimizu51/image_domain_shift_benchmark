# Image Domain Shift Benchmark

This repository provides a reproducible benchmark for analyzing
the robustness of image classification models under distribution shifts.

We focus on common real-world shifts such as resolution degradation,
noise corruption, and illumination changes.

---

## Motivation
Deep learning models often achieve high accuracy on test datasets,
but their performance degrades significantly when the data distribution
changes in real-world environments.
Understanding how and why this degradation occurs is crucial
for deploying reliable vision systems.

---

## Models
- ResNet-18 (current baseline)
- ResNet-50 (planned)
- Vision Transformer (ViT-B/16) (planned)


---

## Distribution Shifts
- Low-resolution inputs
- Gaussian noise corruption
- Illumination changes

---

## Datasets
- CIFAR-10
- CIFAR-100

---

## Goal
The goal of this project is not to propose a new model,
but to systematically analyze failure modes of image classification
models under realistic distribution shifts.

---

## Author
Re  
Undergraduate student, Department of Electrical and Electronic Engineering

---

## Current Status

### Baseline Experiment (Clean Data)

- Dataset: CIFAR-10
- Model: ResNet-18
- Training Epochs: 1
- Batch Size: 128
- Optimizer: Adam (lr=0.001)
- Loss Function: Cross Entropy
- Device: GPU (if available)

### Result
- Training Accuracy: **50.66%**
- Training Loss: 535.739

### Notes
- This experiment serves as a minimal baseline before introducing distribution shifts.
- Evaluation is currently performed on the training set only.
- Test-time robustness under corrupted distributions will be analyzed next.
