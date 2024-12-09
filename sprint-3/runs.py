import os
import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab
from keras.models import load_model

# Load model
model = load_model('model.h5')
print("Model loaded successfully, go for the app")

# Create a main window (named as root)
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognition GUI App")

# Initialize variables
lastx, lasty = None, None
image_number = 0

# Clear the canvas
def clear_widget():
    global cv
    # Clear the canvas
    cv.delete("all")
    instruction_label.config(text="Draw a digit and click 'Recognize Digit' to predict.")


def activate_event(event):
    global lastx, lasty
    # Bind drawing event to canvas
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    # Draw on canvas
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def recognize_digit():
    global image_number
    filename = f'image_{image_number}.png'

    # Get the widget coordinates
    x = root.winfo_rootx() + cv.winfo_x()
    y = root.winfo_rooty() + cv.winfo_y()
    x1 = x + cv.winfo_width()
    y1 = y + cv.winfo_height()

    # Grab the canvas image, crop it, and save as PNG
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    # Read the image in grayscale
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_Black)

    # Apply Otsu's thresholding
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours to detect digit bounding boxes
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for count in contours:
        # Get bounding box and extract the region of interest (ROI)
        x, y, w, h = cv2.boundingRect(count)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Extract the ROI with some padding
        top, bottom, left, right = int(0.05 * th.shape[0]), int(0.05 * th.shape[0]), int(0.05 * th.shape[1]), int(0.05 * th.shape[1])
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th_up[y - top:y + h + bottom, x - left:x + w + right]

        # Resize the ROI to 28x28 for model prediction
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi_resized = roi_resized.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
        roi_resized = roi_resized / 255.0  # Normalize the image

        # Predict the digit using the model
        pred = model.predict(roi_resized)[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        # Draw the prediction text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, font_scale, color, thickness)

    # Show the image with predictions
    cv2.imshow('Predicted Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Add a label for instructions at the top
instruction_label = Label(root, text="Instructions: Draw a digit in the box below and click 'Recognize Digit'.",
                          font=("Helvetica", 12), fg="blue")
instruction_label.grid(row=0, column=0, columnspan=2, pady=10)

# Create a canvas for drawing
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Bind mouse events for drawing
cv.bind('<Button-1>', activate_event)

# Add Buttons
btn_recognize = Button(text="Recognize Digit", command=recognize_digit)
btn_recognize.grid(row=2, column=0, pady=1, padx=1)

btn_clear = Button(text="Clear Widget", command=clear_widget)
btn_clear.grid(row=2, column=1, pady=1, padx=1)

# Main loop
root.mainloop()
