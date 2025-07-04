# 🅿️ Car Parking Detection System

This project detects whether parking spots are occupied or empty using a **custom-trained CNN model** and **manually defined parking spot coordinates**. The system can process images or video and outputs annotated visual results with empty/occupied spot count.
---

## 🔍 Features

- Uses a custom-trained Keras model (`model.h5`) to classify car presence in a region
- Manual parking slot coordinate setup stored in `car_pos.pkl`
- Supports both **images** and **video**
- Annotated output with:
  - Green boxes for **empty** spots
  - Red boxes for **occupied** spots
  - Real-time count display
- Flask backend for easy web integration


## 🛠️ Installation

### 1.Clone the repository
```bash```
git clone https://github.com/M-dorra/car-parking-detection.git
cd car-parking-detection

### 2. Install dependencies : **pip install -r requirements.txt**
### 3. Run Flask server : **python main.py**
### 4. Open the frontend : **http://localhost:5000**

## 📁 Directory Structure
car-parking-detection/
│
├── static/
│
├── main.py                     # Flask server
├── test.py                     # Tests parking model
├── data_collection.py          # Defines parking spots
├── templates/
│   └── index.html              # Frontend template
├── model.h5                    # Trained Keras model
├── car_pos.pkl                 # Saved parking coordinates (list of positions)
├── training_model.ipynb
├── requirements.txt
└── README.md

## ⚙ How It Works

- Predefine Parking Positions :
  -Manually select and save parking spot coordinates using OpenCV click tool (stored in car_pos.pkl)
- Frame Processing
  - For each frame/image, the script:
    - Crops each spot region
    - Resizes and normalizes it
    - Passes it to the loaded model for classification
- Model Inference :
  - Your custom CNN classifies each cropped image as car or no_car.
- Visualization:
  - Colors the parking boxes (red = occupied, green = empty) and overlays count on frame.


