# %%
import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageGrab
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
import tensorflow as tf
from tkinter import *
from tkinter import messagebox, simpledialog
import time
from s3helper import my_logger, S3Helper

# %%
logger = my_logger()
s3_helper = S3Helper()

s3_helper.get_img("digit-recog-example/model.h5")
model_path = 'model.h5'
model = load_model(model_path)
logger.info("Model loaded successfully, ready to use the APP.")
TIME_NOW = time.strftime("%Y%m%d-%H%M%S")

# %%
root = Tk( )
root. resizable (0, 0)
root. title ("Handwritten Digit Recognition GUI App")

# %%
def clear_widget():
    global cv
    cv.delete("all")

def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# %%
def retrain_model(new_data, new_label):
    global model
    new_data = np.array(new_data).reshape(-1, 28, 28, 1)
    new_label = to_categorical(new_label, 10)
    model.fit(new_data, new_label, epochs=1, verbose=0)
    model.save(model_path)
    logger.info("Model retrained and saved.")

# %%
def Recognize_Digit():
    global image_number
    predictions = []
    percentage = []
    #image_number = 0
    filename = f'image_{image_number}.png'
    widget=cv
    # get the widget coordinates
    print(root.winfo_rootx(), widget.winfo_x())
    print(root.winfo_rooty(), widget.winfo_y())
    print(widget.winfo_width(), widget.winfo_height())
    x=root.winfo_rootx() +widget.winfo_x()
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height ()

    print(x, y, x1, y1)
    ImageGrab.grab().crop ((5, 100, 1200, 900)).save (filename )

    image = cv2. imread (filename, cv2. IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2. COLOR_BGR2GRAY )
    ret, th = cv2. threshold(gray, 0,255, cv2.THRESH_BINARY_INV+cv2. THRESH_OTSU)
    contours= cv2. findContours(th, cv2. RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) [0]
    final_pred = None
    img = None
    for cnt in contours:
        x, y,w, h = cv2.boundingRect (cnt)
        cv2. rectangle (image, (x,y), (x+w, y+h), (255, 0, 0) , 1)
        top = int (0.05 * th. shape [0])
        bottom = top
        left = int (0.05 * th. shape[1])
        right = left
        th_up = cv2. copyMakeBorder(th, top, bottom, left, right, cv2. BORDER_REPLICATE)
        roi= th[y-top:y+h+bottom, x-left:x+w+right]
        img = cv2. resize (roi, (28, 28), interpolation=cv2. INTER_AREA)
        img = img. reshape (1, 28,28,1)
        img = img/255.0
        pred = model.predict( [img]) [0]
        logger.info("Before, {}".format(final_pred))
        final_pred = np.argmax (pred)
        logger.info("Mid Validation, {}".format(final_pred))
        data = str (final_pred) +' '+ str(int (max(pred) *100) )+'%'
        font = cv2. FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2. putText (image, data, (x,y-5), font, fontScale, color, thickness)
        img_name = f"{filename}_pred"
        cv2.imwrite(img_name, image)
        s3_helper.upload_file(file_name = img_name, s3_key="digit-recog-example/{}/{}".format(TIME_NOW, img_name))
        break

    logger.info("Final Predcition, {}".format(final_pred))
    is_correct = messagebox.askyesno("Prediction Result", f"Predicted Digit: {final_pred}. Is this correct?")
    if not is_correct:
        new_label = simpledialog.askinteger("Correction", "Enter the correct digit:")
        if new_label is not None:
            retrain_model(img, new_label)
    cv2. imshow ( 'image', image)

# %%
lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
cv.bind('<Button-1>', activate_event)

btn_save = Button(text="Recognize Digit", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)

button_clear = Button(text="Clear Widget", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()

# %%
s3_helper.upload_file(file_name = "digit-recog.log", s3_key="digit-recog-example/{}/digit-recog.log".format(time.strftime("%Y%m%d-%H%M%S")))

# %%


# %%



