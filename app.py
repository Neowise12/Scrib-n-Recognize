from keras.models import load_model

import cv2
import numpy as np
from tkinter import *
import PIL 

from PIL import Image, ImageDraw, ImageGrab


model = load_model('model.h5')

def clear_widget():
    global cv
    cv.delete('all')

def active_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def Recognize_Digit():
    global image_number
    predictions = []
    percentages = []
    filename = f'image_{image_number}.png'
    widget = cv
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()
    PIL.ImageGrab.grab(bbox=(x, y, x1, y1)).save(filename)

    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 28, 28)   
    img = img / 255.0  
    pred = model.predict(img)[0]
    print("Predictions:", pred)
    final_pred = np.argmax(pred)
    print("Final Prediction:", final_pred)

    confidence = 100 * np.max(pred)  
    data = f'Predicted Digit: {final_pred}, Confidence: {confidence:.2f}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0,0,255)  
    thickness = 2
    cv2.putText(image, data, (20, 20), font, font_scale, color, thickness)

    cv2.imshow('Recognized Digit', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image_number += 1


root = Tk()
root.resizable(0, 0)
root.title("Scribe a Digit")

lastx, lasty = None, None
image_number = 0
cv = Canvas(root, width=640, height=480, bg='blue')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
cv.bind('<Button-1>', active_event)
btn_save = Button(text='Recognize', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Clear Screen', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()
