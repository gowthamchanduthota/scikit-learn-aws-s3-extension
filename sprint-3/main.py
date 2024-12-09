
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Initialize the GUI
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

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

        # Process the image for bounding box detection
        image_np = np.array(image)
        _, thresh = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
