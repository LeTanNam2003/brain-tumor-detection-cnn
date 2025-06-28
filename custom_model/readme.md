# Custom CNN Model for Brain Tumor Classification

This repository contains a custom Convolutional Neural Network (CNN) architecture designed for brain tumor classification using MRI images. The model supports four classes: **Normal**, **Meningioma**, **Glioma**, and **Pituitary**.

---

## 1. Network Architecture

The figure below illustrates the architecture of the custom CNN. It includes multiple convolutional layers, ReLU activations, max pooling, and fully connected layers.

![Model Architecture](./assets/network_architecture.png)

*Figure 1: Custom CNN Architecture*

---

## 2. Training Results

The model was trained on an augmented MRI dataset. Below are the training and validation accuracy/loss plots over epochs.

![Training Results](./custom_model/results/images/result_train_iter2.png)

*Figure 2: Training and Validation Accuracy/Loss*

Some key metrics:
- **Best Validation Accuracy**: 94.2%
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 30
- **Learning Rate**: 0.01

---

### 3.1. Requirements

This project requires the following Python libraries:

- os, sys, shutil, io
- tkinter (GUI)
- numpy
- opencv-python (cv2)
- PIL (Pillow)
- pandas
- matplotlib
- seaborn
- torch
- scikit-learn
- tqdm
- psutil
- scipy
- warnings

You can install all required libraries using pip:

```bash
pip install -r requirements.txt
```
