# 🎭 Facial Emotion Recognition (FER) using AffectNet & FER2013

## 🧠 Introduction
This project presents a facial emotion recognition system built using advanced deep learning techniques and trained on two combined benchmark datasets: **AffectNet** and **FER2013**. It aims to classify human facial expressions into seven basic emotion categories:

    😠 Anger, 🤢 Disgust, 😨 Fear, 😄 Happiness, 😢 Sadness, 😲 Surprise, 😐 Neutral

Emotions are a fundamental part of human interaction. Recognizing them through facial expressions enables intelligent systems to become more emotion-aware, enhancing user experience in real-world applications.

Domains like:

    1. Human-Computer Interaction (HCI)
    2. Mental Health Monitoring
    3. E-learning and EdTech
    4. Driver Alertness Systems
    5. Social Robotics
    6. Surveillance and Smart Environments

By leveraging a custom deep CNN inspired by ResNet, Swish activation, and powerful preprocessing strategies, this model achieves high accuracy to **72.25%–73.25%**. The network architecture is optimized for both performance and generalization.

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

## 🧪 Training Details

- **Input Size**: (48, 48, 1)
- **Batch Size**: 256
- **Epochs**: 50
- **Data Augmentation**: Horizontal Flip, shift, Rotation, Zoom,fill
- **Early Stopping** and **Model Checkpointing** and **SWA callback**
- **Learning Rate Schedule**: Warm up + Cosine Decay combined

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