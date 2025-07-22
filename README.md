# 🎭 Facial Emotion Recognition (FER) using AffectNet & FER2013

This project presents a facial emotion recognition system built using advanced deep learning techniques and trained on two combined benchmark datasets: **AffectNet** and **FER2013**. It aims to classify human facial expressions into seven basic emotion categories:

    😠 Anger, 🤢 Disgust, 😨 Fear, 😄 Happiness, 😢 Sadness, 😲 Surprise, 😐 Neutral

Emotions are a fundamental part of human interaction. Recognizing them through facial expressions enables intelligent systems to become more emotion-aware, enhancing user experience in real-world applications.

Domains like:

    Human-Computer Interaction (HCI)

    Mental Health Monitoring

    E-learning and EdTech

    Driver Alertness Systems

    Social Robotics

    Surveillance and Smart Environments

By leveraging a custom deep CNN inspired by ResNet, Swish activation, and powerful preprocessing strategies, this model achieves high accuracy to 72.25%–73.25%. The network architecture is optimized for both performance and generalization.

---

## 📂 Datasets Used

### 1. [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- 48x48 grayscale images
- 30k+ images
- Emotions: 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)

### 2. [AffectNet](https://www.kaggle.com/datasets/mstjebashazida/affectnet)
- 1 million facial images from the internet
- Images vary in pose, lighting, ethnicity
- Emotions: 7 (merge Contempt into disgust)

---

## 🧠 Model Architecture

### Custom CNN (ResNet-inspired)

Input Layer: (48, 48, 1)

────────────────── Residual Block 1 ─────────────────────────
Conv2D (32, 3x3) → BN → Swish                         
Conv2D (32, 3x3) → BN → Swish                         
Conv2D (1x1, stride=2) on shortcut → SpatialDropout2D 
Output: (48, 48, 32)                                  

───────────────── Residual Block 2 ──────────────────────────
Conv2D (64, 3x3, stride=2) → BN → Swish               
Conv2D (64, 3x3) → BN → Swish                         
Conv2D (1x1, stride=2) on shortcut → SpatialDropout2D 
Output: (24, 24, 64)                                  

───────────────── Residual Block 3 ──────────────────────────
Conv2D (128, 3x3, stride=2) → BN → Swish              
Conv2D (128, 3x3) → BN → Swish                        
Conv2D (1x1, stride=2) on shortcut → SpatialDropout2D 
Output: (12, 12, 128)                                 

───────────────── Residual Block 4 ──────────────────────────
Conv2D (256, 3x3, stride=2) → BN → Swish              
Conv2D (256, 3x3) → BN → Swish                        
Conv2D (1x1, stride=2) on shortcut → SpatialDropout2D 
Output: (6, 6, 256)                                   

───────────────── Residual Block 5 ──────────────────────────
Conv2D (512, 3x3, stride=2) → BN → Swish              
Conv2D (512, 3x3) → BN → Swish                        
Conv2D (1x1, stride=2) on shortcut → SpatialDropout2D 
Output: (3, 3, 512)                                   

GlobalAveragePooling2D → Output: (512)

BN (512)

Fully Connected Dense Layers:
→ Dense(256) → BN → Swish → Dropout(0.5)
→ Dense(128) → BN → Swish → Dropout(0.5)
→ Dense(64)  → BN → Swish → Dropout(0.5)
→ Dense(32)  → BN → Swish → Dropout(0.5)

Temperature Scaling Layer: 0.7

Output Layer: Dense(num_classes=7, activation='softmax')

---

## 🧪 Training Details

- **Input Size**: (48, 48, 1)
- **Batch Size**: 256
- **Epochs**: 50–60
- **Data Augmentation**: Horizontal Flip, shift, Rotation, Zoom,fill
- **Early Stopping** and **Model Checkpointing** and **SWA callback**
- **Learning Rate Schedule**: Warm up + Cosine Decay combined

---

## 🏁 Features

- 📦 Supports both **FER2013** and **AffectNet** datasets
- 🧪 Includes **training**, **validation**, **testing** and **sample test** pipeline

---

## 📊 Evaluation Metrics

- Accuracy, Loss (cross-entropy)
- Confusion Matrix
- classification report 
- training vs validation charts for **accuracy**,**loss**, **precision**, **recall**, **auc**
- samples from internet

---

## 🧠 Highlights

| Component                 | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| Residual Connections      | Deep supervision and gradient flow                   |
| Swish Activation          | Outperforms ReLU in many emotion recognition tasks   |
| SpatialDropout2D          | Regularizes conv layers, improves generalization     |
| Batch Normalization       | After every Conv and Dense layer                     |
| GlobalAveragePooling      | More robust than Flatten for spatial aggregation     |
| Temperature Scaling (0.7) | Softens logits to improve calibration and confidence |