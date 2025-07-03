# Potato Plant Disease Classification Workflow

This document outlines the step-by-step workflow implemented in the notebook for classifying potato plant diseases using deep learning feature extraction and a Random Forest classifier.

---

## Workflow Steps

### 1. **Import Libraries**
Essential libraries for data handling, image processing, machine learning, and visualization are imported:
- PyTorch, torchvision, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn, numpy.

### 2. **(Optional) Mount Google Drive**
If running in Google Colab, Google Drive is mounted to access datasets.

### 3. **Set Data Path**
The path to the potato leaf image dataset is specified.

### 4. **Data Loading and Preprocessing**
- Images are loaded using `torchvision.datasets.ImageFolder`.
- Images are resized, normalized, and converted to tensors.
- DataLoader is used to batch and iterate over the dataset.

### 5. **Feature Extraction with ResNet50**
- A pretrained ResNet50 model is loaded, with the final fully connected layer replaced by an identity layer.
- The model is set to evaluation mode.
- Deep features are extracted for all images and stored along with their labels.

### 6. **Class Balancing with SMOTE**
- SMOTE (Synthetic Minority Over-sampling Technique) is applied to the extracted features to balance the class distribution.

### 7. **Train/Test Split**
- The balanced dataset is split into training and test sets, stratified by class.

### 8. **Random Forest Training and Cross-Validation**
- A Random Forest classifier is initialized with class weighting.
- 5-fold cross-validation is performed on the training set.
- The classifier is trained on the full training set.

### 9. **Model Evaluation**
- Predictions are made on the test set.
- Confusion matrix and classification report are generated.
- Test accuracy is calculated.

### 10. **Visualization**
- The confusion matrix is visualized using a heatmap for better interpretability.

### 11. **Model Saving**
- The ResNet feature extractor weights and the trained Random Forest model are saved for future inference.

---

## Summary

This workflow enables robust classification of potato plant diseases by combining deep learning for feature extraction and classical machine learning for classification, while addressing class imbalance with SMOTE. The notebook provides both quantitative and visual evaluation
