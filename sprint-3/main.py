<<<<<<< HEAD
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model("model.h5")

=======
# Initialize the GUI
>>>>>>> 106b4da (Update main.py)
class DigitRecognizerApp:
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

<<<<<<< HEAD
        # Canvas for drawing
        self.canvas = tk.Canvas(self.root, width=300, height=300, bg="white", cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons for actions
        button_frame = tk.Frame(self.root)
        button_frame.pack()
=======
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
>>>>>>> 106b4da (Update main.py)

        self.recognize_button = tk.Button(button_frame, text="Recognize", command=self.recognize, width=10)
        self.recognize_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas, width=10)
        self.clear_button.pack(side="right", padx=5)

    def draw(self, event):
        """Draw circles on the canvas where the mouse moves."""
        r = 8  # Radius of the drawing brush
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="")

    def clear_canvas(self):
        """Clear the canvas."""
        self.canvas.delete("all")
        self.status_label.config(text="Status: Canvas cleared", fg="blue")

    def recognize(self):
        """Recognize the digit(s) drawn on the canvas."""
        # Capture the canvas content as an image
        x, y = self.root.winfo_rootx() + self.canvas.winfo_x(), self.root.winfo_rooty() + self.canvas.winfo_y()
        x1, y1 = x + self.canvas.winfo_width(), y + self.canvas.winfo_height()
        image = ImageGrab.grab(bbox=(x, y, x1, y1)).convert("L")

        # Convert the image to a NumPy array for processing
        image_np = np.array(image)
        _, thresh = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

<<<<<<< HEAD
        # Process each detected contour
=======
        if contours:
            self.status_label.config(text="Status: Recognizing digit...", fg="orange")

        # Recognize each digit in bounding boxes
>>>>>>> 106b4da (Update main.py)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            digit_image = image_np[y:y + h, x:x + w]

            # Normalize and resize the digit image
            digit_image = cv2.resize(digit_image, (28, 28))
            digit_image = digit_image / 255.0
            digit_image = digit_image.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = model.predict(digit_image, verbose=0)
            digit = np.argmax(prediction)

            # Draw bounding box and label
            self.canvas.create_rectangle(x, y, x + w, y + h, outline="red", width=2)
            self.canvas.create_text(x + w // 2, y - 10, text=str(digit), fill="blue", font=("Arial", 12, "bold"))

        self.status_label.config(text="Status: Recognition complete", fg="green")

# Initialize the Tkinter root and app
<<<<<<< HEAD
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
=======
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()

>>>>>>> 106b4da (Update main.py)
