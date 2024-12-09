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

        # Add a frame for buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        # Button to recognize digits
        self.recognize_button = tk.Button(root, text="Recognize", command=self.recognize)
        self.recognize_button.pack()

        # Button to clear the canvas
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        # Button to quit the app
        self.quit_button = tk.Button(
            self.button_frame, text="Quit", command=root.quit
        )
        self.quit_button.grid(row=0, column=3, padx=10)

    def draw(self, event):
        # Draw on the canvas
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")

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

            # Model prediction
        prediction = model.predict(image)
        digit = np.argmax(prediction)

            # Display the result
        messagebox.showinfo("Recognition Result", f"Predicted Digit: {digit}")

        # except Exception as e:
        #     messagebox.showerror("Error", f"An error occurred during recognition: {e}")

        self.status_label.config(text="Status: Recognition complete", fg="green")

# Initialize the Tkinter root and app
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
