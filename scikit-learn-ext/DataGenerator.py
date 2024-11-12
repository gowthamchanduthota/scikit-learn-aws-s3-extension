import numpy as np
import tensorflow as tf
from PIL import Image
from s3helper import *

class S3DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, bucket_name, file_keys, labels_age, labels_gender, batch_size, img_size=(128, 128), n_channels=1, shuffle=True, aws_access_key_id=None, aws_secret_access_key=None):
        self.bucket_name = bucket_name
        self.file_keys = file_keys
        self.labels_age = labels_age
        self.labels_gender = labels_gender
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.s3_client = S3Helper(datasets_bucket=bucket_name, credentials={
            "AWS_ACCESS_KEY_ID": aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key
        })
        self.indexes = np.arange(len(self.file_keys))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_keys) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_file_keys = [self.file_keys[k] for k in batch_indexes]
        X, y_age, y_gender = self.__data_generation(batch_file_keys, batch_indexes)
        return X, (y_gender, y_age)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_keys))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_file_keys, batch_indexes):
        X = np.zeros((self.batch_size, *self.img_size, self.n_channels), dtype=np.float32)
        y_age = np.zeros((self.batch_size,), dtype=np.float32)
        y_gender = np.zeros((self.batch_size,), dtype=np.int32)

        for i, (file_key, idx) in enumerate(zip(batch_file_keys, batch_indexes)):
            # Load and preprocess image from S3
            image_data = self.s3_client.get_img(f"UTKFace/{file_key}")
            image = image_data.resize(self.img_size).convert('L')
            X[i] = np.expand_dims(np.array(image) / 255.0, axis=-1)  # Normalize and expand dims

            # Load labels
            y_age[i] = self.labels_age[idx]
            y_gender[i] = self.labels_gender[idx]

        return X, y_age, y_gender
