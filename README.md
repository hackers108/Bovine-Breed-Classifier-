# Indian Bovine Breeds Classifier

**Deployed Project**: [Hugging Face Space](https://huggingface.co/spaces/hacker108/cow-breed-classifier)

---

## Project Overview

This project aims to classify Indian bovine breeds from images using deep learning techniques.
It includes a Gradio-based web application with two key features:

* **Prediction**: Upload an image and receive the top three predicted breeds with probabilities.
* **Labeling**: Upload and label new images to contribute to dataset expansion and continuous improvement.

The work covers the entire machine learning pipeline—dataset preparation, model training, inference, interface development, and deployment.

---

## Key Achievements

* Achieved **80% validation accuracy** across **12 Indian bovine breeds**.
* Designed and deployed a **Gradio web application** for inference and labeling.
* Deployed the solution on **Hugging Face Spaces**.
* Integrated with **GitHub** for reproducibility and model updates.

---

## Dataset Collection and Processing

* **Source**: Kaggle dataset containing approximately 6,000 images labeled by breed.
* **Data cleaning**: Removed noisy and incomplete images, addressed cases with multiple or occluded cows.
* **Augmentation**: Applied random rotations, flips, resizing, and normalization to improve robustness.
* **Segmentation**: Explored **SAM ViT-B** for cow segmentation, later switched to **YOLO** for efficiency.

**Dataset structure:**

```
data/
 ├── train/<breed_name>/*.jpg
 └── test/<breed_name>/*.jpg
```

---

## Model Development and Training

* **Approach**: Fine-tuned pre-trained models rather than training from scratch.
* **Architectures evaluated**: ResNet50, DenseNet121, MobileNetV2, InceptionV3, EfficientNet-B3.
* **Selected model**: **EfficientNet-B3**, balancing accuracy and efficiency.
* **Preprocessing**: Image resizing, horizontal flips, rotations, tensor conversion, and ImageNet normalization.
* **Training**: Conducted on a Kaggle T4 GPU with model checkpoints saved after validation improvements.
* **Result**: 80% validation accuracy achieved.

---

## Inference

* Implemented an inference notebook for testing the trained model.
* Provides **top-3 predictions with probabilities**.
* Verified performance on unseen images.

---

## Gradio Web Application

* **Prediction Tab**: Upload an image and receive classification results.
* **Labeling Tab**: Upload and label images for dataset growth.
* **Download option**: Users can export labeled images as a structured ZIP archive.
* **Dynamic model loading from GitHub** ensures a lightweight and up-to-date application.

---

## Deployment

* Hosted on **Hugging Face Spaces** for public access.
* Integrated with **GitHub** for version control and future updates.
* Provides both immediate inference and community-driven dataset contribution.

---

## Repository Structure

```
Bovine-Breed-Classifier/
 ├── notebooks/         # Training and inference notebooks
 ├── app/               # Gradio app and requirements
 ├── best_model.pth     # Trained model (managed via Git LFS or downloadable)
 ├── README.md
 ├── LICENSE            # Apache 2.0
```

GitHub Repository: [https://github.com/hackers108/Bovine-Breed-Classifier-](https://github.com/hackers108/Bovine-Breed-Classifier-)

---

## Key Learnings and Challenges

* Large segmentation models like **SAM ViT-B** demand significant computational resources and are less practical for smaller-scale work.
* **YOLO** proved to be a more efficient option for object detection and segmentation in this context.
* Fine-tuning pre-trained models provided significant benefits compared to training from scratch.
* Language models assisted with code scaffolding, but domain expertise in machine learning was critical for achieving results.

---

## Tools and Technologies

* **Deep Learning**: PyTorch, EfficientNet, YOLO, SAM ViT-B
* **Data Handling**: Preprocessing, augmentation, dataset structuring
* **Deployment**: Gradio, Hugging Face Spaces
* **Version Control**: Git, GitHub, Git LFS
* **Scripting**: Python for model management, labeling interface, and inference workflows

---

## Summary

This project delivers a complete machine learning pipeline—from dataset curation and model training to web deployment—achieving reliable bovine breed classification with 80% accuracy.
It provides a functional, publicly accessible tool and a framework for future improvements through community contributions.
