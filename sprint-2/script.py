import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
import random
from tensorflow.keras.preprocessing import image
import logging
import os


DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"

# s3 = boto3.client('s3')
# bucket = s3.Bucket(DATASETS_BUCKET)

def my_logger():
    curr_script = os.path.basename(__file__)
    logging.basicConfig(
        filename="{}.log".format(curr_script),
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

logger = my_logger()

class S3Helper:

    def __init__(self, datasets_bucket, outputs_bucket, credentials = None):
        self.s3 = boto3.client('s3')

        if credentials:
            session = boto3.Session(
                aws_access_key_id=credentials.get("AWS_ACCESS_KEY_ID", None),
                aws_secret_access_key=credentials.get("AWS_SECRET_ACCESS_KEY", None),
            )
            self.s3 = session.client('s3')
            logger.info("Using Credentials: ", credentials)
            # update boto client to use credentials
        self.datasets_bucket = datasets_bucket
        self.outputs_bucket = outputs_bucket

    def get_img(self, path, show = False):
        """Downloads an image from S3 and optionally displays it."""
        logger.info(f"Getting image from path: {path}")
        filename = path.split("/")[-1]
        try:
            self.s3.download_file(
                Bucket=self.datasets_bucket,
                Key=path,
                Filename=filename
            )
            img = image.load_img(filename)
            if show:
                imgrd = mpimg.imread(filename)
                plt.imshow(imgrd)
                plt.show()
            return img
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return None

    def upload_file(self, file_name, s3_key = None, bucket = None):
        """Uploads a file to S3."""
        if not s3_key:
            s3_key = os.path.basename(file_name)

        if not bucket:
            bucket = self.outputs_bucket

        try:
            self.s3.upload_file(
                Bucket=bucket,
                Key=s3_key,
                Filename=file_name
            )
            logger.info(f"File uploaded at: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")

    def list_objects(self, bucket, path = ""):
        """Lists files in the specified S3 bucket."""
        try:
            resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=path)
            files = [obj['Key'] for obj in resp.get('Contents', []) if obj['Size']]
            logger.info(f"Found {len(files)} files.")
            return files
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []

    def get_frac(self, frac, path = "", random_state = 42, download_files = False):
        """Fetch a fraction of files from S3."""
        logger.info(f"Fetching {frac * 100}% of files from path: {path}")
        obj_list = self.list_objects(self.datasets_bucket, path)
        if not obj_list:
            return []

        dataset_len = len(obj_list)
        rand_list = random.sample(obj_list, int(dataset_len * frac))
        
        if download_files:
            for it in rand_list:
                try:
                    self.s3.download_file(
                        Bucket=self.datasets_bucket,
                        Key=it,
                        Filename=it
                    )
                    logger.info(f"Downloaded file: {it}")
                except Exception as e:
                    logger.error(f"Error downloading {it}: {e}")

        return rand_list

def main():

    s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

    # img = s3_helper.get_img("sample/dog1.jpeg")
    # logger.info(img)
    path = "/Users/gowtham/WorkSpace/UNT/SE/datasets/UTKFace"
    dir_list = os.listdir(path)
    dataset_len = len(dir_list)
    logger.info(f"Total dataset size: {dataset_len}")
    fracc = int(dataset_len * 0.1)
    logger.info(f"Selected {fracc} files for processing")
    rand_lists = random.sample(dir_list, fracc)
    logger.info(f"Sampled list length: {len(rand_lists)}")

    # Uncomment if needed to upload files
    for it in rand_lists:
        try:
            s3_helper.upload_file(f"UTKFace/{it}", f"UTKFace/{it}", bucket=DATASETS_BUCKET)
        except Exception as e:
            logger.error(f"Error uploading {it}: {e}")

    files = s3_helper.list_objects(DATASETS_BUCKET)
    logger.info(f"Files in the dataset bucket: {len(files)}")

if __name__ == "__main__":
    main()

