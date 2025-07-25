import tkinter as tk
from tkinter import filedialog
from tkinter import *

from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"


def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1.json","model_weights1.h5")



EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = facec.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        label1.configure(foreground="#011638", text="No face detected.")
        print("No face detected.")
        return

    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y + h, x:x + w]

            # validate detected face quality
            if np.std(fc) < 20:
                label1.configure(foreground="#011638", text="Detected face seems invalid.")
                print("Low contrast face area")
                return

            roi = cv2.resize(fc, (48, 48))
            roi = roi[np.newaxis, :, :, np.newaxis]

            pred_idx = np.argmax(model.predict(roi))
            pred = EMOTIONS_LIST[pred_idx]
            print("Predicted Emotion is " + pred)
            label1.configure(foreground="#011638", text=pred)
            return
    except Exception as e:
        print("Error during prediction:", e)
        label1.configure(foreground="#011638", text="Error in detecting emotion.")





def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()