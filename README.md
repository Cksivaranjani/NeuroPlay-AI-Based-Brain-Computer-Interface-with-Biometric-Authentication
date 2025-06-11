# 🧠 NeuroPlay: AI-Based Brain-Computer Interface with Biometric Authentication

This project presents *NeuroPlay, an AI-powered Brain-Computer Interface (BCI) system that allows users to interact with a digital platform using brain signals. It also integrates **face and voice recognition* for secure and accessible user authentication. The system supports multiple modules including a brain-controlled game, thought-to-speech conversion, mood-based music, and cognitive alerting.

---

## 🚀 Features

- EEG signal classification using Support Vector Machine (SVM)
- Face and voice-based user authentication (registration & login)
- Brain-controlled maze game based on mental state
- Thought-to-speech module for non-verbal communication
- Mood-based music player
- Real-time alerts based on brain activity patterns

---

## 📌 Objective

To create an accessible, intelligent platform where users can interact with technology through brain signals and AI, eliminating the need for physical input devices—especially beneficial for users with disabilities.

---

## 🧠 Mental State Labels

- *0.0* → Relaxed or Idle  
- *1.0* → Stressed  
- *2.0* → Concentrated

---

## 📊 Model Performance

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 94.96% |
| Precision    | 0.95   |
| Recall       | 0.95   |
| F1-Score     | 0.95   |

Sample Predictions:
- Input 1 → Predicted: 2.0 → Concentrated  
- Input 2 → Predicted: 1.0 → Stressed  
- Input 3 → Predicted: 0.0 → Relaxed  

---

## 🛠 Tech Stack

- *Frontend*: HTML, CSS, JavaScript  
- *Backend*: Python (Flask)  
- *AI/ML*: Scikit-learn (SVM), NumPy, SciPy  
- *Biometric Authentication*: OpenCV, Dlib, Resemblyzer  
- *Database*: SQLite  
- *EEG Dataset*: Pre-recorded brainwave signals representing cognitive states

---

## 🔐 Modules

- Face & Voice-based Login/Registration  
- BCI Signal Processing and AI Model Training  
- Brain-Controlled Maze Game  
- Thought-to-Speech Generator  
- Mood-Based Music Player  
- Cognitive Alerts and Insights

---

## 📥 Setup Instructions

```bash
git clone https://github.com/your-username/NeuroPlay-BCI.git
cd NeuroPlay-BCI
pip install -r requirements.txt
python app.py
