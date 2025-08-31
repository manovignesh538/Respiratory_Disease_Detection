# Respiratory_Disease_Detection

A deep learning-based project to detect respiratory diseases, specifically focusing on binary classification between **COVID-19** and **Normal** cases using chest X-ray images.

---

## 🧠 Model Overview
 
- **Model Type**: Custom CNN (Convolutional Neural Network)
- **Task**: Binary Classification (COVID-19 vs. Normal)
- **Architecture Highlights**:
  - 4 Convolutional Blocks
  - MaxPooling and Dropout layers to reduce overfitting
  - Fully Connected Dense layers
  - Sigmoid activation function for binary output
- **Total Layers**: 17
- **Trainable Parameters**: 1,457,537

---

## 🗂 Dataset

- **Training Path**: `C:\Users\vigne\data_new\Dataset\Train`
- **Validation Path**: `C:\Users\vigne\data_new\Dataset\Val`
- **Images**:
  - Training: 288 images (2 classes)
  - Validation: 60 images (2 classes)

---

## 🧪 Data Augmentation

To improve model generalization, the following augmentation techniques were used:

- Rescaling
- Rotation (up to 30 degrees)
- Width/Height shifting
- Shearing
- Zooming
- Horizontal flipping

---

## 🏗 Model Architecture (Sequential)

| Layer Type     | Output Shape        | Parameters |
|----------------|---------------------|------------|
| Conv2D (32)     | (222, 222, 32)       | 896        |
| Conv2D (64)     | (220, 220, 64)       | 18,496     |
| MaxPooling2D    | (110, 110, 64)       | 0          |
| Dropout         | -                   | 0          |
| Conv2D (64)     | (108, 108, 64)       | 36,928     |
| MaxPooling2D    | (54, 54, 64)         | 0          |
| Dropout         | -                   | 0          |
| Conv2D (128)    | (52, 52, 128)        | 73,856     |
| MaxPooling2D    | (26, 26, 128)        | 0          |
| Dropout         | -                   | 0          |
| Conv2D (128)    | (24, 24, 128)        | 147,584    |
| MaxPooling2D    | (12, 12, 128)        | 0          |
| Dropout         | -                   | 0          |
| Flatten         | (18432)              | 0          |
| Dense (64)      | (64)                 | 1,179,712  |
| Dropout         | -                   | 0          |
| Dense (1)       | (1)                  | 65         |

---

## ⚙️ Model Compilation

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
