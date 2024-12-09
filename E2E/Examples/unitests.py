import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from keras.models import Sequential
from tkinter import messagebox, simpledialog
import cv2
from guiscript import *

class TestDigitRecognition(unittest.TestCase):

    @patch('s3helper.S3Helper')
    @patch('keras.models.load_model')
    def setUp(self, mock_load_model, mock_s3_helper):
        self.mock_s3_helper = mock_s3_helper.return_value
        self.mock_model = mock_load_model.return_value
        self.mock_model.predict.return_value = np.array([[0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.model_path = 'model.h5'

    @patch('PIL.ImageGrab.grab')
    @patch('cv2.resize')
    @patch('cv2.threshold')
    @patch('cv2.findContours', return_value=([], None))
    @patch('cv2.cvtColor')
    @patch('cv2.imread')
    def test_recognize_digit(self, mock_imread, mock_cvtcolor, mock_find_contours, mock_threshold, mock_resize, mock_grab):
        mock_grab.return_value.crop.return_value.save = MagicMock()
        mock_imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_threshold.return_value = (None, np.zeros((480, 640), dtype=np.uint8))
        mock_resize.return_value = np.zeros((28, 28), dtype=np.uint8)

        with patch('tkinter.messagebox.askyesno', return_value=False):
            with patch('tkinter.simpledialog.askinteger', return_value=3):
                Recognize_Digit()

                self.mock_s3_helper.upload_file.assert_called()
                self.mock_model.predict.assert_called()
                self.assertEqual(self.mock_model.predict.call_count, 1)

    @patch('keras.models.load_model')
    def test_retrain_model(self, mock_load_model):
        mock_model = mock_load_model.return_value
        mock_model.fit = MagicMock()
        mock_model.save = MagicMock()

        new_data = np.zeros((28, 28), dtype=np.uint8)
        new_label = 3

        retrain_model([new_data], new_label)

        mock_model.fit.assert_called_once()
        mock_model.save.assert_called_once_with(self.model_path)

    @patch('tkinter.Canvas.delete')
    def test_clear_widget(self, mock_delete):
        global cv
        cv = MagicMock()
        clear_widget()
        cv.delete.assert_called_once_with("all")

    @patch('tkinter.Canvas.bind')
    def test_activate_event(self, mock_bind):
        event = MagicMock()
        event.x = 100
        event.y = 150

        activate_event(event)
        mock_bind.assert_called_once_with('<B1-Motion>', draw_lines)

    @patch('tkinter.Canvas.create_line')
    def test_draw_lines(self, mock_create_line):
        global lastx, lasty
        lastx, lasty = 50, 60
        event = MagicMock()
        event.x = 100
        event.y = 150

        draw_lines(event)
        mock_create_line.assert_called_once()

if __name__ == '__main__':
    unittest.main()
