# Malaria Detection from Cell Images Using Deep Learning

## Objective

This project aims to develop and evaluate deep learning models capable of accurately distinguishing between malaria-parasitized and uninfected red blood cells using cell image data.

## Dataset

The dataset originates from the **NIH Malaria Dataset**, containing 27,558 labeled images of parasitized and uninfected cells.
*   [Link to Dataset Source (Optional but recommended if available)]

## Methodology

### 1. Data Preprocessing

To ensure data consistency and prepare images for model training, the following preprocessing steps were applied:
*   Resized all images to `150x150` pixels.
*   Normalized pixel values to a range of `[0, 1]`.
*   Applied data augmentation techniques (random zooming, flipping, rotation) to increase the effective dataset size and mitigate overfitting.

### 2. Model Architectures Explored

Two primary deep learning approaches were investigated:

*   **Custom Convolutional Neural Network (CNN):** A CNN built from scratch featuring convolutional layers for feature extraction, pooling layers for down-sampling, and fully connected layers for final classification.
*   **Transfer Learning (VGG19):** Utilized the pre-trained VGG19 model (trained on ImageNet). Base convolutional layers were frozen, and only the final classification layers were retrained on the malaria dataset to leverage pre-learned features.

### 3. Training & Validation

*   The dataset was split into 70% for training and 30% for validation for both models.

## Results & Observations

### Custom CNN

*   **Training Accuracy:** 94.4%
*   **Validation Accuracy:** 95.28%
*   **Validation Loss:** Showed signs of slight overfitting in later epochs.
*   **Observations:**
    *   The custom CNN architecture effectively learned distinguishing features, achieving high classification accuracy.
    *   Data augmentation proved beneficial for model generalization.
    *   Overfitting indicates a need for stronger regularization.

### Transfer Learning (VGG19)

*   **Training Accuracy:** 91.25%
*   **Validation Accuracy:** 90.36%
*   **Validation Loss:** Remained relatively stable but performance was lower than the custom CNN.
*   **Observations:**
    *   Leveraging VGG19's pre-trained weights reduced training time compared to the custom CNN.
    *   Despite stability, the model didn't achieve the same peak accuracy as the custom CNN on this specific task, possibly due to feature mismatch or model complexity.

## Discussion

The custom-built CNN demonstrated superior performance in terms of validation accuracy for this specific malaria detection task compared to using transfer learning with VGG19 out-of-the-box (with frozen base layers). However, the custom CNN showed indications of overfitting, suggesting further refinement is needed. Transfer learning offered faster training convergence but requires further investigation (e.g., fine-tuning, different architectures) to potentially surpass the custom model.

## Planned Improvements & Future Work

To further enhance model performance and robustness:

*   **Address Overfitting:** Implement stronger regularization techniques (e.g., L2 regularization, increased dropout rates) in the custom CNN.
*   **Hyperparameter Tuning:** Systematically tune parameters like learning rate, batch size, and dropout rates for both models.
*   **Explore Deeper/Different Architectures:**
    *   Investigate deeper custom CNN architectures.
    *   Experiment with other pre-trained models for transfer learning (e.g., ResNet50, EfficientNet).
*   **Fine-Tuning:** Selectively unfreeze and retrain some layers of the pre-trained models (like VGG19) to better adapt them to the malaria dataset.
*   **Robust Evaluation:**
    *   Implement k-fold cross-validation for more reliable performance estimation.
    *   Utilize additional evaluation metrics beyond accuracy (Precision, Recall, F1-score, AUC).
*   **Detailed Comparison:** Conduct further analysis comparing the refined custom CNN against optimized transfer learning approaches.

## Getting Started (Optional - Add if code is runnable)

### Requirements
*   Python 3.x
*   TensorFlow / Keras
*   NumPy
*   Matplotlib (for visualization)
*   [List any other specific libraries]

### Installation
```bash
# Example installation commands (adjust as needed)
pip install tensorflow numpy matplotlib

# Example command to run training or prediction script
python train_model.py --data_path /path/to/dataset
