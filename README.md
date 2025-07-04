# 🅿️ Car Parking Detection System

This project is a full-stack web application that detects and counts **occupied and free parking spaces** from a video feed or an image. It uses a **custom-trained CNN model** to classify each predefined parking region as `"car"` or `"no car"`. The interface is built using **Flask**, and the video processing is done with **OpenCV** and **TensorFlow/Keras**.

## 🔍 Features

- Real-time parking detection from a video stream
- Uses a custom-trained Keras model (`model.h5`) to classify car presence in a region
- Manual parking slot coordinate setup stored in `car_pos.pkl`
- Supports both **images** and **videos**
- Annotated output with:
  - Green boxes for **empty** spots
  - Red boxes for **occupied** spots
  - Real-time count display
- Flask backend for easy web integration


## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/M-dorra/car-parking-detection.git
cd car-parking-detection
```
### 2. Install dependencies :
Make sure you have Python installed. Then, install the required Python packages:

```bash
pip install -r requirements.txt
```
### 3. Run Flask server :

```bash
python main.py
```
### 4. Open the frontend : **http://localhost:5000**


## ⚙️ How It Works
1. Manual Spot Definition
  - Parking spots are manually defined and saved using OpenCV mouse clicks
  - Saved to car_pos.pkl using the tool define_parking_spots.py
2. Frame Processing
  - For each frame in the video:
  - Crop each defined region (ROI)
  - Resize to match model input (e.g., 48x48)
  - Normalize pixel values
  - Pass to CNN model for prediction
3. Classification
  - CNN returns label "car" or "no_car" for each ROI
4. Visualization
  - Draws colored boxes (green/red)
  - Counts empty and occupied spots
  - Sends results to web interface and API


![Screenshot 2025-07-04 160944](https://github.com/user-attachments/assets/7bb389ce-1d33-4fa3-b2b0-7f3418810105)

![Screenshot 2025-07-04 150448](https://github.com/user-attachments/assets/46aebe97-f209-4160-a238-37aa9a5d0a95)
