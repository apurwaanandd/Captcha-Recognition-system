# 🤖 CAPTCHA Recognition System

## 📘 Overview

This project is a **CAPTCHA Recognition Web Application** developed using **Python, Flask, and Deep Learning**. It aims to automate the process of CAPTCHA verification by recognizing and solving CAPTCHA images using a **trained Convolutional Neural Network (CNN)** model. The system provides a user-friendly web interface where users can upload CAPTCHA images and instantly receive predicted results.

---

## 🎯 Objective

To develop a simple and efficient system that can:
- Recognize and solve CAPTCHA images
- Help users bypass visual CAPTCHA challenges (for research/demo purposes)
- Demonstrate the power of AI and OCR (Optical Character Recognition) using deep learning models

---

## ✨ Features

- 🧠 Deep learning-based character recognition using a trained `.h5` model
- 🖼️ Image upload functionality via a clean web interface
- 🧾 Real-time CAPTCHA prediction
- 🔍 Supports alphanumeric CAPTCHA recognition
- 🧪 Flask-based backend for quick deployment and testing

---

## 🧰 Tech Stack

- **Python** – Core programming language
- **Flask** – Lightweight web framework for the backend
- **TensorFlow / Keras** – Deep learning framework for model training and prediction
- **HTML/CSS** – Web front-end interface

---

## 📂 Project Structure

📦 Captcha-Recognition-system/ ├── app.py # Main Flask server ├── captcha_model.h5 # Trained CNN model ├── captcha_reader.py # Utility script to read/predict CAPTCHA ├── index.html # Web interface ├── 2b827.png, 2g7nm.png # Sample CAPTCHA images ├── tempCodeRunnerFile.py # Temporary test file └── README.md # Project documentation


> ⚠️ **Note:** The original image dataset used for training and testing was too large to upload here. Only **two sample CAPTCHA images** (`2b827.png`, `2g7nm.png`) are included for demonstration purposes. You can add more during testing as needed.

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.x
- Flask
- TensorFlow / Keras
- OpenCV
- Numpy

### 🛠️ Installation Steps

1. Clone the github repository
  
3. Install Dependencies
pip install -r requirements.txt
(If requirements.txt is missing, install manually: Flask, tensorflow, keras, opencv-python)

4. Run the Flask App
python app.py
Open the Web App Visit http://127.0.0.1:5000 in your browser.
📸 Sample Output

5. Upload a CAPTCHA image → Get the predicted text instantly
Works with alphanumeric characters
📥 Drive Backup

🔗 Google Drive Link (Backup)
🔮 Future Enhancements

5. Support for distorted or noisy CAPTCHAs
Add model retraining module from uploaded images
Extend support to multi-line CAPTCHA images
Integrate OCR techniques for printed CAPTCHA styles
📚 References

6. Flask Documentation
TensorFlow Guide
Keras OCR
OpenCV Python

Google Drive Link:
https://drive.google.com/file/d/1WKvmiT2Rdur11wkeI2SOj74BtAIsTXvt/view?usp=sharing


**Clone the Repository**
```bash
git clone https://github.com/apurwaanandd/Captcha-Recognition-system.git
cd Captcha-Recognition-system
