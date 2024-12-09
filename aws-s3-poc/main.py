import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import boto3
import random
import logging
import os
import time

# Constants
DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"

def setup_logger():
    """
    Configures and returns a logger with both file and console output.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler
    curr_script = os.path.basename(__file__)
    file_handler = logging.FileHandler(f"{curr_script}.log")
    file_formatter = logging.Formatter(
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()

class S3Helper:
    def __init__(self, datasets_bucket, outputs_bucket, credentials=None):
        """
        Initialize the S3Helper with optional AWS credentials.
        """
        self.datasets_bucket = datasets_bucket
        self.outputs_bucket = outputs_bucket

        if credentials:
            session = boto3.Session(
                aws_access_key_id=credentials.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=credentials.get("AWS_SECRET_ACCESS_KEY"),
            )
            self.s3 = session.client("s3")
            logger.info("Using provided AWS credentials.")
        else:
            self.s3 = boto3.client("s3")

    def get_img(self, path, show=False):
        """
        Download an image from S3 and return it as a PIL image object.
        Optionally display the image.
        """
        filename = os.path.basename(path)
        self.s3.download_file(self.datasets_bucket, path, filename)
        logger.info("Downloaded image: %s", filename)

        img = image.load_img(filename)
        if show:
            plt.imshow(mpimg.imread(filename))
            plt.axis("off")
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
        pages = paginator.paginate(Bucket=bucket, Prefix=path)

        files = [obj["Key"] for page in pages for obj in page.get("Contents", []) if obj["Size"]]
        logger.info("Found %d objects in %s/%s", len(files), bucket, path)
        return files

    def get_frac(self, frac, path="", random_seed=42, download_files=False):
        """
        Retrieve a fraction of files from S3 at the specified path.
        """
        obj_list = self.list_objects(self.datasets_bucket, path)
        sample_count = max(1, int(len(obj_list) * frac))  # Ensure at least one file is chosen

        random.seed(random_seed)
        sampled_files = random.sample(obj_list, sample_count)
        logger.info("Selected %d/%d files (%.2f%%) from %s", sample_count, len(obj_list), frac * 100, path)

        if download_files:
            for key in sampled_files:
                local_dir = os.path.dirname(key)
                if local_dir and not os.path.exists(local_dir):
                    os.makedirs(local_dir)

                self.s3.download_file(self.datasets_bucket, key, key)
            logger.info("Downloaded sampled files to %s", path)

        return sampled_files

def main():
    """
    Main function to demonstrate S3Helper functionality.
    """
    s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

    # Retrieve and display an image
    img = s3_helper.get_img("sample/dog1.jpeg", show=False)
    logger.info("Retrieved image: %s", type(img))

    # List files in a directory
    files = s3_helper.list_objects(DATASETS_BUCKET, "UTKFace")
    logger.info("Listed %d files in UTKFace: %s", len(files), files[:5])  # Display first 5 files

    # Retrieve a fraction of files
    frac_files = s3_helper.get_frac(0.1, path="UTKFace", random_seed=42, download_files=True)
    logger.info("Sampled %d files: %s", len(frac_files), frac_files[:5])  # Display first 5 sampled files

    # Upload a log file
    log_file_path = f"{os.path.basename(__file__)}.log"
    if os.path.exists(log_file_path):
        s3_helper.upload_file(
            file_name=log_file_path,
            s3_key=f"test-s3-poc/{time.strftime('%Y%m%d-%H%M%S')}/{log_file_path}"
        )
    else:
        logger.warning("Log file not found: %s", log_file_path)

if __name__ == "__main__":
    main()
