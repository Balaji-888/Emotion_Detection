

# Emotion_Detection
This project is a deep learning-powered emotion detection system that analyzes facial expressions from images and accurately classifies them into predefined emotional categories. Designed with a user-friendly interface using Tkinter, it allows users to upload images and instantly see the predicted emotion label.


## 📌 Project Highlights

- 🔍 Detects emotions from facial expressions in images.
- 📷 Uses OpenCV's Haar Cascade for face detection.
- 🧠 Predicts one of 7 emotions: "Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise".
- 💡 Easy-to-use Python GUI application using Tkinter.
- ✅ Offline execution – no need for internet after setup.
- 👥 It supports Single face detection and Multiple face detection in a single image.u
---



🚀 Features

- Detects multiple faces and predicts each face's emotion individually.

- Draws bounding boxes with emotion labels on detected faces.

- Supports JPG, PNG, and other common image formats.

- Works offline, no internet connection required.

- Easy-to-use GUI for uploading and testing images.





## 🖥️ GUI Preview

Upload an image with a visible human face and click "Detect Emotion". The predicted emotion will be displayed abovethe image.

---


🛠️ Installation & Setup

1. Clone the repository

  git clone https://github.com/your-username/Emotion_Detection.git
  cd Emotion_Detection

2. Create a virtual environment (recommended)

  python -m venv emotion_env
  emotion_env\Scripts\activate   # For Windows
  source emotion_env/bin/activate  # For Mac/Linux

3. Install dependencies

  pip install -r requirements.txt

4. Download Haar Cascade file (if not already included)
  Haarcascade_frontalface_default.xml



---



▶️ Usage

1. Run the GUI:
     python gui.py

2. Click Upload Image in the GUI.


3. Select an image with a face/faces.


4. Click Detect Emotion.


5. The detected faces will be highlighted with predicted emotions.

---



## 🧠 Model Details

- Model architecture: CNN
- Trained using: FER-2013 dataset
- Libraries used: TensorFlow, Keras, OpenCV, NumPy

---


📷 Sample Output



1️⃣ Single Faces Detected

<img width="993" height="781" alt="Screenshot 2025-08-09 120533" src="https://github.com/user-attachments/assets/d6629494-ce7e-46cd-9545-d59a11e6a7e0" />








2️⃣ Multiple Face Detected


<img width="995" height="776" alt="Screenshot 2025-08-09 114847" src="https://github.com/user-attachments/assets/95bf355c-cf5d-43ab-8858-162945448263" />



---




📝 Dependencies
 Main libraries used in this project:

- tensorflow

- keras

- opencv-python

- numpy

- Pillow

- scikit-learn

- tkinter


---



  
## 📂 File Structure
```

Emotion_Detection/
├── gui.py # Main GUI application
├── model_creation.ipynb # Model training notebook (optional)
├── model_a1.json # Model architecture (CNN)
├── model_weights1.h5 # Trained model weights
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── requirements.txt # Python dependencies
└── README.md # Project documentation

```



📌 Future Improvements

- Add real-time webcam emotion detection

- Deploy as a web app using Streamlit

- Integrate with voice sentiment analysis for combined emotion detection.

---



## 👨‍💻 Author

**Balaji M**  
📧 Email: [balajim26114@gmail.com](mailto:balajim26114@gmail.com)  
🔗 GitHub: [@Balaji-888](https://github.com/Balaji-888)


----



⭐ Contribute & Support
If you find this project helpful or interesting:

⭐ Star the repository

🍴 Fork it

📬 Share with others

Feel free to raise issues or suggestions. Contributions are always welcome!
