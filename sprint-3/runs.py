import os
import cv2
import numpy as np
from tkinter import *
<<<<<<< HEAD
from PIL import ImageGrab, Image
=======
<<<<<<< HEAD
from PIL import Image, ImageDraw, ImageGrab
>>>>>>> 4149365 (Update runs.py)
from keras.models import load_model

# Load the model
model = load_model('model.h5')
<<<<<<< HEAD
print("Model loaded successfully. Ready to run the application.")
=======
print("Model loaded successfully. Ready to use the app.")

# Create a main window (root)
=======
from PIL import ImageGrab
from keras.models import load_model

# Load model
model = load_model('model.h5')
print("Model loaded successfully, go for the app")

# Create a main window (named as root)
>>>>>>> 44d07f8 (Update runs.py)
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognition GUI App")
>>>>>>> 4149365 (Update runs.py)

# Initialize variables
last_x, last_y = None, None
image_number = 0

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")

# Function to start drawing
def activate_drawing(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

# Function to draw on the canvas
def draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    canvas.create_line(last_x, last_y, x, y, width=8, fill="black", capstyle=ROUND, smooth=TRUE, splinesteps=12)
    last_x, last_y = x, y

# Function to recognize the digit
def recognize_digit():
    global image_number
    filename = f"digit_{image_number}.png"
    image_number += 1

<<<<<<< HEAD
    # Get canvas coordinates and save as image
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    ImageGrab.grab(bbox=(x, y, x1, y1)).save(filename)
=======
    # Get widget coordinates
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Capture the canvas area and save as an image
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
>>>>>>> 4149365 (Update runs.py)

    # Process the image
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

<<<<<<< HEAD
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = thresh[y:y + h, x:x + w]

        # Resize and normalize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.reshape(1, 28, 28, 1) / 255.0
=======
    for cnt in contours:
        # Get bounding box and extract ROI
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        roi = th[y - top:y + h + bottom, x - left:x + w + right]
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    # Display the image with predictions
    cv2.imshow('Prediction', image)

# Add a label for instructions at the top
instruction_label = Label(root, text="Instructions: Draw a digit in the box below and click 'Recognize Digit'.",
                          font=("Helvetica", 12), fg="blue")
instruction_label.grid(row=0, column=0, columnspan=2, pady=10)
>>>>>>> 4149365 (Update runs.py)

        # Predict the digit
        prediction = model.predict([roi])[0]
        predicted_digit = np.argmax(prediction)
        confidence = int(max(prediction) * 100)

        # Display prediction on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{predicted_digit} {confidence}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Show the processed image with predictions
    cv2.imshow("Recognized Digits", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create the main application window
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognition")

# Create a drawing canvas
canvas = Canvas(root, width=640, height=480, bg="white")
canvas.grid(row=0, column=0, pady=10, columnspan=2)
canvas.bind("<Button-1>", activate_drawing)
canvas.bind("<B1-Motion>", draw)

# Add buttons for actions
recognize_button = Button(root, text="Recognize Digit", command=recognize_digit, width=15)
recognize_button.grid(row=1, column=0, pady=10, padx=10)

clear_button = Button(root, text="Clear Canvas", command=clear_canvas, width=15)
clear_button.grid(row=1, column=1, pady=10, padx=10)

# Start the application
root.mainloop()
<<<<<<< HEAD
=======


# Add Buttons
btn_recognize = Button(text="Recognize Digit", command=recognize_digit)
btn_recognize.grid(row=2, column=0, pady=1, padx=1)

btn_clear = Button(text="Clear Widget", command=clear_widget)
btn_clear.grid(row=2, column=1, pady=1, padx=1)

# Main loop
root.mainloop()
>>>>>>> 4149365 (Update runs.py)
