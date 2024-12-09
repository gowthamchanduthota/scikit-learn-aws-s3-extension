import unittest
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Dropout, Input, Dense, BatchNormalization, Flatten, Conv2D,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from s3helper import S3Helper
from DataGenerator import S3DataGenerator
from ModelWrapper import S3ModelWrapper

DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"

class Sprint3Testing(unittest.TestCase):
    def setUp(self):
        self.s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)
        filenames = self.s3_helper.list_objects(DATASETS_BUCKET, "UTKFace")
        filenames = [os.path.basename(file) for file in filenames]

        np.random.seed(10)
        np.random.shuffle(filenames)

        age_labels, gender_labels = zip(*[(int(fname.split('_')[0]), int(fname.split('_')[1])) for fname in filenames])
        df = pd.DataFrame({'image': filenames, 'age': age_labels, 'gender': gender_labels})

        self.gender_dict = {0: "Male", 1: "Female"}
        df = df.astype({'age': 'float32', 'gender': 'int32'})
        self.train, self.test = train_test_split(df, test_size=0.985, random_state=42)

        self.batch_size = 5
        self.img_size = (128, 128)
        self.n_channels = 1

        self.x_train = np.array([np.array(self.s3_helper.get_img(f"UTKFace/{file}", greyscale=True).resize(self.img_size, Image.LANCZOS))
                                 for file in self.train['image']]).reshape(-1, 128, 128, 1) / 255.0

        self.y_gender = self.train['gender'].values
        self.y_age = self.train['age'].values

        self.model = self.build_model()
        self.model = S3ModelWrapper(self.model)
        self.model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=[['accuracy'], ['mae']])

    def build_model(self):
        input_size = (128, 128, 1)
        inputs = Input(shape=input_size)

        X = Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_uniform')(inputs)
        X = BatchNormalization()(X)
        X = MaxPooling2D((3, 3))(X)

        X = Conv2D(128, (4, 5), activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)

        X = Conv2D(256, (3, 3), activation='relu')(X)
        X = MaxPooling2D((2, 2))(X)

        X = Flatten()(X)
        dense_shared = Dense(256, activation='relu')(X)

        output_gender = Dense(1, activation='sigmoid', name='gender_output')(Dropout(0.4)(dense_shared))
        output_age = Dense(1, activation='relu', name='age_output')(Dropout(0.4)(Dense(128, activation='relu')(dense_shared)))

        return Model(inputs=inputs, outputs=[output_gender, output_age])

    def test_wrapper_fit(self):
        model_history = self.model.fit(self.x_train, [self.y_gender, self.y_age], batch_size=10, epochs=1, validation_split=0.1)
        self.assertIsNotNone(model_history.history)

    def test_wrapper_predict(self):
        index = 28
        pred = self.model.predict(self.x_train[index:index+1])
        pred_gender = self.gender_dict[round(pred[0][0][0])]
        pred_age = round(pred[1][0][0])
        self.assertGreaterEqual(pred_age, 0)
        self.assertLessEqual(pred_age, 150)

    def test_fit_with_dg(self):
        train_generator = S3DataGenerator(
            bucket_name=DATASETS_BUCKET,
            file_keys=self.train['image'].tolist(),
            labels_age=self.train['age'].tolist(),
            labels_gender=self.train['gender'].tolist(),
            batch_size=self.batch_size,
            img_size=self.img_size,
            n_channels=self.n_channels,
            shuffle=True
        )
        model_history = self.model.fit(train_generator, epochs=1, verbose=1)
        self.assertIsNotNone(model_history.history)

    def test_predict_with_dg(self):
        index = 3
        file_key = self.train.iloc[index]['image']
        true_age = self.train.iloc[index]['age']
        true_gender = self.gender_dict[self.train.iloc[index]['gender']]

        img = self.s3_helper.get_img(f"UTKFace/{file_key}", greyscale=True)
        img_array = np.expand_dims(np.array(img.resize(self.img_size, Image.LANCZOS)) / 255.0, axis=[0, -1])

        pred = self.model.predict(img_array)
        pred_gender = self.gender_dict[round(pred[0][0][0])]
        pred_age = round(pred[1][0][0])

        self.assertGreaterEqual(pred_age, 0)
        self.assertLessEqual(pred_age, 150)

if __name__ == '__main__':
    unittest.main()
