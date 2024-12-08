import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
import random
from tensorflow.keras.preprocessing import image
import logging
import os
import time

DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"

<<<<<<< HEAD
def configure_logger(log_file="digit-recog.log"):
    """Configures and returns a logger."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger(__name__)

logger = configure_logger()

class S3Helper:
    def __init__(self, datasets_bucket=DATASETS_BUCKET, outputs_bucket=OUTPUTS_BUCKET, credentials=None):
        """Initializes the S3Helper class."""
=======
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# logger = logging.getLogger()

def my_logger():
    curr_script = os.path.basename(__file__)
    logging.basicConfig(
        filename="{}.log".format("digit-recog"),
        level=logging.INFO,
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    return logger


class S3Helper:

    def __init__(self, datasets_bucket=DATASETS_BUCKET, outputs_bucket=OUTPUTS_BUCKET, credentials=None):
>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)
        if credentials:
            session = boto3.Session(
                aws_access_key_id=credentials.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=credentials.get("AWS_SECRET_ACCESS_KEY"),
            )
            self.s3 = session.client('s3')
<<<<<<< HEAD
            logger.info("Using provided AWS credentials.")
=======
            print("Using provided AWS credentials.")
>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)
        else:
            self.s3 = boto3.client('s3')

        self.datasets_bucket = datasets_bucket
        self.outputs_bucket = outputs_bucket

<<<<<<< HEAD
    def get_img(self, s3_key, show=False, greyscale=False):
        """Downloads an image from S3 and optionally displays it."""
        local_filename = os.path.basename(s3_key)
        self.s3.download_file(self.outputs_bucket, s3_key, local_filename)
        logger.info("Downloaded file from S3: %s", s3_key)

        if show:
            img = image.load_img(local_filename, color_mode="grayscale" if greyscale else "rgb")
            plt.imshow(mpimg.imread(local_filename))
            plt.axis('off')
            plt.show()

        return local_filename

    def upload_file(self, file_name, s3_key=None, bucket=None):
        """Uploads a file to the specified S3 bucket."""
        s3_key = s3_key or os.path.basename(file_name)
        bucket = bucket or self.outputs_bucket
        self.s3.upload_file(file_name, bucket, s3_key)
        logger.info("File uploaded to S3: %s", s3_key)

    def list_objects(self, bucket, prefix=""):
        """Lists objects in an S3 bucket with the specified prefix."""
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        return [obj['Key'] for page in pages for obj in page.get('Contents', [])]

    def get_frac(self, frac, prefix="", random_seed=42, download_files=False):
        """Retrieves a random fraction of objects from an S3 bucket."""
        obj_list = self.list_objects(self.datasets_bucket, prefix)
=======
    def get_img(self, path, show=False, greyscale=False):
        filename = os.path.basename(path)
        self.s3.download_file(self.outputs_bucket, path, filename)


        if show:
            img = image.load_img(filename, color_mode="grayscale" if greyscale else "rgb")
            plt.imshow(mpimg.imread(filename))
            plt.axis('off')
            plt.show()

        # os.remove(filename)
        # return img

    def upload_file(self, file_name, s3_key=None, bucket=None):
        s3_key = s3_key or os.path.basename(file_name)
        bucket = bucket or self.outputs_bucket
        self.s3.upload_file(file_name, bucket, s3_key)
        print("File uploaded at: %s", s3_key)

    def list_objects(self, bucket, path=""):
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=path)
        return [obj['Key'] for page in pages for obj in page.get('Contents', [])]

    def get_frac(self, frac, path="", random_seed=42, download_files=False):
        obj_list = self.list_objects(self.datasets_bucket, path)
>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)
        frac_len = int(len(obj_list) * frac)

        random.seed(random_seed)
        rand_list = random.sample(obj_list, frac_len)

        if download_files:
<<<<<<< HEAD
            for obj in rand_list:
                local_path = os.path.dirname(obj)
                os.makedirs(local_path, exist_ok=True)
                self.s3.download_file(self.datasets_bucket, obj, obj)
            logger.info("Downloaded %d files from S3 to %s", frac_len, prefix)
=======
            for it in rand_list:
                local_dir = os.path.dirname(it)
                os.makedirs(local_dir, exist_ok=True)
                self.s3.download_file(self.datasets_bucket, it, it)
            print("Downloaded %d files to %s", frac_len, path)
>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)

        return rand_list

def main():
    s3_helper = S3Helper()

    # Retrieve and display an image
<<<<<<< HEAD
    img_filename = s3_helper.get_img("sample/dog1.jpeg", show=True)
    logger.info("Image info: %s", img_filename)

    # Upload a log file
    log_file_path = "digit-recog.log"
    s3_key = f"test-s3-poc/{time.strftime('%Y%m%d-%H%M%S')}/digit-recog.log"
=======
    img = s3_helper.get_img("sample/dog1.jpeg", show=True)
    print("Image info: %s, Type: %s", img, type(img))

    # Upload a log file
    log_file_path = "main.py.log"
    s3_key = f"test-s3-poc/{time.strftime('%Y%m%d-%H%M%S')}/main.py.log"
>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)
    s3_helper.upload_file(log_file_path, s3_key)

if __name__ == "__main__":
    main()

<<<<<<< HEAD

=======
>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)
