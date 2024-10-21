import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
import random
from tensorflow.keras.preprocessing import image
import logging
import os
import time

# Constants
DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"

# Logger configuration
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Helper:
    def __init__(self, datasets_bucket=DATASETS_BUCKET, outputs_bucket=OUTPUTS_BUCKET, credentials=None):
<<<<<<< HEAD
=======
        if credentials:
            session = boto3.Session(
                aws_access_key_id=credentials.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=credentials.get("AWS_SECRET_ACCESS_KEY"),
            )
            self.s3 = session.client('s3')
            logger.info("Using provided AWS credentials.")
        else:
            self.s3 = boto3.client('s3')

>>>>>>> cf67fc8 (Added modelWrapper)
        self.datasets_bucket = datasets_bucket
        self.outputs_bucket = outputs_bucket

        # Create an S3 client
        self.s3 = boto3.client('s3') if not credentials else boto3.Session(
            aws_access_key_id=credentials["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=credentials["AWS_SECRET_ACCESS_KEY"]
        ).client('s3')

        logger.info("Initialized S3Helper.")

    def get_img(self, path, show=False, greyscale=False):
        """
        Download an image from S3, optionally display it, and return as a PIL image object.
        """
        filename = os.path.basename(path)
        self.s3.download_file(self.datasets_bucket, path, filename)
        logger.info("Downloaded image: %s", filename)

        img = image.load_img(filename, color_mode="grayscale" if greyscale else "rgb")

        if show:
            plt.imshow(img if greyscale else mpimg.imread(filename))
            plt.axis('off')
            plt.show()

        os.remove(filename)
        logger.info("Removed local file: %s", filename)
        return img

    def upload_file(self, file_name, s3_key=None, bucket=None):
        """
        Upload a local file to an S3 bucket.
        """
        s3_key = s3_key or os.path.basename(file_name)
        bucket = bucket or self.outputs_bucket

        self.s3.upload_file(file_name, bucket, s3_key)
        logger.info("File uploaded to %s/%s", bucket, s3_key)

    def list_objects(self, bucket=None, path=""):
        """
        List objects in the specified S3 bucket and path.
        """
        bucket = bucket or self.datasets_bucket
        paginator = self.s3.get_paginator("list_objects_v2")

        keys = [
            obj["Key"] for page in paginator.paginate(Bucket=bucket, Prefix=path)
            for obj in page.get("Contents", [])
        ]
        logger.info("Listed %d objects in %s/%s", len(keys), bucket, path)
        return keys

    def get_frac(self, frac, path="", random_seed=42, download_files=False):
        """
        Retrieve a fraction of files from S3 at the specified path.
        """
        obj_list = self.list_objects(self.datasets_bucket, path)
<<<<<<< HEAD
        frac_len = max(1, int(len(obj_list) * frac))  # Ensure at least one object is chosen

        random.seed(random_seed)
        sampled_keys = random.sample(obj_list, frac_len)
        logger.info("Sampled %d files (%.2f%%) from %s", frac_len, frac * 100, path)

        if download_files:
            for key in sampled_keys:
                local_path = os.path.dirname(key)
                os.makedirs(local_path, exist_ok=True)
                self.s3.download_file(self.datasets_bucket, key, key)
            logger.info("Downloaded sampled files to local path: %s", path)

        return sampled_keys
=======
        frac_len = int(len(obj_list) * frac)

        random.seed(random_seed)
        rand_list = random.sample(obj_list, frac_len)

        if download_files:
            for it in rand_list:
                local_dir = os.path.dirname(it)
                os.makedirs(local_dir, exist_ok=True)
                self.s3.download_file(self.datasets_bucket, it, it)
            logger.info("Downloaded %d files to %s", frac_len, path)

        return rand_list
>>>>>>> cf67fc8 (Added modelWrapper)

def main():
    """
    Main function to demonstrate S3Helper functionality.
    """
    s3_helper = S3Helper()

    # Example: Retrieve and display an image
    img_path = "sample/dog1.jpeg"
    img = s3_helper.get_img(img_path, show=True)
    logger.info("Retrieved image: %s, Type: %s", img_path, type(img))

    # Example: Upload a log file to S3
    log_file_path = "main.py.log"
    s3_key = f"test-s3-poc/{time.strftime('%Y%m%d-%H%M%S')}/main.py.log"
    if os.path.exists(log_file_path):
        s3_helper.upload_file(log_file_path, s3_key)
    else:
        logger.warning("Log file does not exist: %s", log_file_path)

if __name__ == "__main__":
    main()
