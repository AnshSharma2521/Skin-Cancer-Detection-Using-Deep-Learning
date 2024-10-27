# Melanoma Cancer Detection

This repository provides code and resources for detecting melanoma cancer using deep learning. The primary goal of this project is to develop a model that accurately classifies skin lesions as benign or malignant, aiding in the early detection of melanoma.

---

## Table of Contents
1. [Project Background](#project-background)
2. [Dataset Description](#dataset-description)
3. [Model Architecture](#model-architecture)
4. [Methodology](#methodology)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Error Analysis](#error-analysis)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [References](#references)
12. [License](#license)

---

## Project Background

Melanoma is a serious type of skin cancer, and early detection is crucial for effective treatment. Traditional detection methods rely on manual analysis and may vary in accuracy. This project uses machine learning to develop an accessible and automated melanoma detection system that can support clinical diagnosis and help identify high-risk cases earlier.

---

## Dataset Description

The dataset used in this project is sourced from Kaggle and is available at [Melanoma Cancer Detection on Kaggle](https://www.kaggle.com/code/nirmalgaud/melanoma-cancer-detection). It includes images of skin lesions categorized into benign and malignant classes, along with metadata to provide additional information for model training.

- **Source**: [Kaggle - Melanoma Cancer Detection](https://www.kaggle.com/code/nirmalgaud/melanoma-cancer-detection)
- **Data Attributes**:
  - **Images**: High-quality images of skin lesions, primarily in `.jpeg` format.
  - **Metadata**: Contains information on age, gender, anatomical site, and lesion dimensions.
- **Data Volume**: Thousands of labeled images, with a mix of benign and malignant samples.
- **Purpose**: This dataset supports machine learning models designed to differentiate benign from malignant melanoma by analyzing visual and metadata cues in skin lesions.

### Data Preprocessing
- **Image Resizing**: Standardized dimensions across the dataset to facilitate uniform input to the model.
- **Normalization**: Normalized pixel values for better convergence during training.
- **Data Augmentation**: Techniques like rotation, scaling, and flipping were applied to increase diversity and improve model robustness.

---

## Model Architecture

The following model architectures were explored in this project:

# Model Architectures

This section provides a detailed layer-by-layer architecture description of each model used in the project: VGG16, ResNet50, and the Custom CNN. Each model has been fine-tuned for melanoma detection and optimized for effective feature extraction.

---

## VGG16 Architecture Layers

The VGG16 model is a Convolutional Neural Network (CNN) with 16 weight layers, designed by the Visual Geometry Group (VGG) at the University of Oxford. It consists of 13 convolutional layers, 5 max-pooling layers, and 3 fully-connected layers, primarily using 3x3 convolution filters and ReLU activations.

### Summary of Layers

| Layer Type           | Filters/Units | Kernel Size | Activation | Output Shape       |
|----------------------|---------------|-------------|------------|--------------------|
| Input                | -             | -           | -          | 224x224x3          |
| Conv 1               | 64            | 3x3         | ReLU       | 224x224x64         |
| Conv 2               | 64            | 3x3         | ReLU       | 224x224x64         |
| Max Pooling          | -             | 2x2         | -          | 112x112x64         |
| Conv 3               | 128           | 3x3         | ReLU       | 112x112x128        |
| Conv 4               | 128           | 3x3         | ReLU       | 112x112x128        |
| Max Pooling          | -             | 2x2         | -          | 56x56x128          |
| Conv 5               | 256           | 3x3         | ReLU       | 56x56x256          |
| Conv 6               | 256           | 3x3         | ReLU       | 56x56x256          |
| Conv 7               | 256           | 3x3         | ReLU       | 56x56x256          |
| Max Pooling          | -             | 2x2         | -          | 28x28x256          |
| Conv 8               | 512           | 3x3         | ReLU       | 28x28x512          |
| Conv 9               | 512           | 3x3         | ReLU       | 28x28x512          |
| Conv 10              | 512           | 3x3         | ReLU       | 28x28x512          |
| Max Pooling          | -             | 2x2         | -          | 14x14x512          |
| Conv 11              | 512           | 3x3         | ReLU       | 14x14x512          |
| Conv 12              | 512           | 3x3         | ReLU       | 14x14x512          |
| Conv 13              | 512           | 3x3         | ReLU       | 14x14x512          |
| Max Pooling          | -             | 2x2         | -          | 7x7x512            |
| Fully Connected 1    | 4096          | -           | ReLU       | 1x1x4096           |
| Fully Connected 2    | 4096          | -           | ReLU       | 1x1x4096           |
| Fully Connected (Output) | 1     | -           | Sigmoid     | 1                   |

---

## ResNet50 Architecture Layers

The ResNet50 model is a 50-layer residual network that incorporates skip connections, which help to prevent the vanishing gradient problem, enabling deeper architectures with high accuracy.

### Summary of Layers

| Layer Type           | Filters/Units | Kernel Size | Activation | Output Shape       |
|----------------------|---------------|-------------|------------|--------------------|
| Input                | -             | -           | -          | 224x224x3          |
| Conv 1               | 64            | 7x7         | ReLU       | 112x112x64         |
| Max Pooling          | -             | 3x3         | -          | 56x56x64           |
| Conv Block 1         | 64            | 3x3         | ReLU       | 56x56x256          |
| Conv Block 2         | 128           | 3x3         | ReLU       | 28x28x512          |
| Conv Block 3         | 256           | 3x3         | ReLU       | 14x14x1024         |
| Conv Block 4         | 512           | 3x3         | ReLU       | 7x7x2048           |
| Global Avg Pooling   | -             | -           | -          | 1x1x2048           |
| Fully Connected (Output) | 1     | -           | Sigmoid     | 1                   |

---

## Custom CNN Architecture Layers

The Custom CNN model is a simplified convolutional neural network created to experiment with different network architectures and to optimize for computational efficiency.

### Summary of Layers

| Layer Type           | Filters/Units | Kernel Size | Activation | Output Shape       |
|----------------------|---------------|-------------|------------|--------------------|
| Input                | -             | -           | -          | 128x128x3          |
| Conv 1               | 32            | 3x3         | ReLU       | 128x128x32         |
| Conv 2               | 32            | 3x3         | ReLU       | 128x128x32         |
| Max Pooling          | -             | 2x2         | -          | 64x64x32           |
| Conv 3               | 64            | 3x3         | ReLU       | 64x64x64           |
| Conv 4               | 64            | 3x3         | ReLU       | 64x64x64           |
| Max Pooling          | -             | 2x2         | -          | 32x32x64           |
| Conv 5               | 128           | 3x3         | ReLU       | 32x32x128          |
| Max Pooling          | -             | 2x2         | -          | 16x16x128          |
| Flatten              | -             | -           | -          | 32768              |
| Fully Connected 1    | 128           | -           | ReLU       | 128                |
| Dropout              | -             | -           | -          | 128                |
| Fully Connected (Output) | 1     | -           | Sigmoid     | 1                   |

---

Each model architecture is structured to enhance melanoma classification accuracy, providing a range of depth and complexity based on model selection.


---

## Methodology

The methodology applies CNN-based approaches with additional preprocessing and model tuning steps to ensure high classification accuracy. The key stages include:

### 1. Data Preprocessing
   - Steps as detailed in the [Dataset Description](#dataset-description).

### 2. Training Strategy
   - Optimizer: Adam optimizer with a learning rate scheduler.
   - Loss Function: Binary cross-entropy, suitable for binary classification tasks.
   - Batch Size and Epochs: Experimented with various values to find optimal performance.

### 3. Evaluation and Deployment
   - Metrics: Evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.
   - Deployment: Saved trained models as `.h5` files and provided inference scripts.

---

## Installation

To set up the project environment, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/melanoma-cancer-detection.git
cd melanoma-cancer-detection
pip install -r requirements.txt
```

## Author
**Ansh Sharma**
