# 🌿 AgriVision — Plant Disease Detector

An AI-powered web application that detects plant diseases from leaf photographs 
in real time. Upload a photo of a leaf and the app instantly identifies the disease, 
severity, and recommends both organic and chemical treatments.

## 🔍 Overview

Trained on the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) 
— 54,000+ leaf images across 15 disease and healthy classes covering crops including 
Tomato, Potato, Bell Pepper

## ✨ Features

- 📸 Upload any leaf photo (JPG, PNG, WEBP)
- 🧠 Instant inference using a quantized TFLite model (93% accuracy)
- 🔴 Disease severity rating — High / Medium / Low
- 💊 Organic and chemical treatment recommendations
- 📊 Top-5 confidence bar chart for transparency
- ⚡ Lightweight and fast — runs entirely in the browser via Streamlit

## 🛠️ Tech Stack

- **Model:** MobileNetV2-based CNN using TensorFlow/Keras Transfer Learning
- **Dataset:** PlantVillage (15 classes, 54,000+ images)
- **App:** Python · Streamlit · TensorFlow Lite · Plotly
- **Deployment:** Streamlit Community Cloud (free)

## 🚀 Try it live

👉 [Click here to open the app](#) ← replace with your Streamlit URL
