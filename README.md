# 🧠 Moody – Real-Time Emotion & Gesture Recognition with Sound Reactions

Moody is an AI-powered system that detects facial emotions and hand gestures through a webcam — and reacts by playing custom sounds.  
It is designed for streamers, creative projects, AI demos and real-time interactive applications.

Moody works in three phases:

- **Setup Phase** – Calibrate your face & assign sounds  
- **Detection Phase** – Real-time emotion + gesture recognition  
- **Reaction Phase** – Moody plays the mapped sound  


---

## 🚀 Features

- DeepFace-based emotion recognition  
- MediaPipe-based gesture detection (e.g., thumbs up)  
- Custom sound assignment using an integrated browser (myinstants.com)  
- Personalized emotion baseline calibration  

---

## 📦 Project Structure
```mermaid
flowchart TD
    A[MoodyStream]

    A --> MAIN[main.py - Entry point]
    A --> MAIN[requirements.txt - necessary requirements]
    A --> DET[detection/ - Emotion + gesture detection pipeline]
    A --> SET[setup/ - Face calibration + sound setup wizard]
    A --> SND[sounds/ - Sound files + behavior-to-sound mapping]
    A --> GUI[gui/ - Optional UI components]
    A --> UTL[utils/ - Helper modules: JSON + settings]
    A --> WEB[web/ - Browser logic for sound downloading]
```
---

## 🛠️ How Moody Works

### Setup Phase

**Face Setup:**  
Moody guides you through multiple emotions (e.g., happy, sad, fear, surprise, neutral).  
For each emotion, the webcam captures several samples while DeepFace and FaceMesh extract features.  
Moody then computes:

- a personalized neutral baseline  
- emotion-specific statistics  
- dynamic thresholds for distinguishing real expressions  
- and a small personalized classifier

All captured profiles and example snapshots are saved and used to make the live emotion detection more stable and accurate.


**Sound Setup:**  
Moody opens a built-in browser window with myinstants.com.  
When you click on sounds, they are downloaded automatically and you assign them to behaviors (happy, angry, thumbsup, etc.).  
The mapping is saved so Moody knows which sound to trigger.

---

## 🎥 Detection Phase

Once setup is finished:

- The webcam runs continuously  
- Emotions are detected with DeepFace  
- Gestures are detected with MediaPipe  
- A behavior is selected  
- The corresponding sound is played immediately  

---

## 🔊 Sound System

All downloaded sound files are stored in:
sounds/sound_cache/

The mapping between behaviors and sounds is stored in:
sounds/sound_map.json

---

## ▶️ How to Run

Install dependencies:
```
pip install -r requirements.txt
```
Start Moody:
```
python main.py
```
The setup wizard will automatically start on first run.

---

## 🧪 Developer Notes

- Add new gestures in the detectors folder  
- Change emotion/gesture priority in the recognition logic  
- All path definitions and global settings are in the utils folder  
- JSON data is safely handled by the json manager module  

---

## 📚 Requirements

- Python 3.9+  
- Webcam  
- macOS / Windows / Linux  
- Internet connection only needed for sound setup  

---

## 📝 License

MIT License (or define your own)
