import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model (MNIST CNN model)
model = load_model('mnist_cnn_model.h5')  # Make sure you have the model file

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        # Add a title label
        self.title_label = tk.Label(root, text="Draw a digit below and click Recognize", font=("Helvetica", 14))
        self.title_label.pack()

        # Create a canvas for drawing
        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Add a status label
        self.status_label = tk.Label(root, text="Status: Waiting for input", font=("Helvetica", 12), fg="green")
        self.status_label.pack()

        # Button to recognize digits
        self.recognize_button = tk.Button(root, text="Recognize", command=self.recognize)
        self.recognize_button.pack()

        # Button to clear the canvas
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # Button to quit the app
        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack()

    def draw(self, event):
        # Draw on the canvas
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.status_label.config(text="Status: Canvas cleared", fg="blue")

    def recognize(self):
        # Capture canvas content as an image
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        image = ImageGrab.grab().crop((x, y, x1, y1)).convert("L")

        # Preprocess the image
        image = image.resize((28, 28), Image.ANTIALIAS)  # Resize to 28x28 for the model
        image = np.array(image)
        image = cv2.bitwise_not(image)  # Invert colors to match MNIST (white digits on black background)
        image = image / 255.0  # Normalize to range [0, 1]
        image = image.reshape(1, 28, 28, 1)  # Reshape for the model input

        if contours:
            self.status_label.config(text="Status: Recognizing digit...", fg="orange")

        # Recognize each digit in bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            digit_image = image_np[y:y+h, x:x+w]
            digit_image = cv2.resize(digit_image, (28, 28))
            digit_image = digit_image / 255.0
            digit_image = digit_image.reshape(1, 28, 28, 1)
            prediction = model.predict(digit_image)
            digit = np.argmax(prediction)

        # Display result on the canvas
        messagebox.showinfo("Recognition Result", f"Predicted Digit: {digit}")

        # Optionally, draw the predicted digit on the canvas
        self.canvas.create_text(150, 150, text=str(digit), font=("Helvetica", 24), fill="blue")

        self.status_label.config(text="Status: Recognition complete", fg="green")

# Initialize the Tkinter root and app
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()

