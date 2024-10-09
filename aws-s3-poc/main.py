import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
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
        print(path)
        filename = path.split("/")[-1]
        self.s3.download_file(
            Bucket = self.datasets_bucket,
            Key = path,
            Filename = filename
        )
        img = image.load_img(filename)
        if show:
            imgrd = mpimg.imread(filename)
            plt.imshow(imgrd)
            plt.show()
        return img

    # Update this fn to be a decorator call fn
    def upload_file(self, path):
        filename = path.split("/")[-1]
        self.s3.upload_file(
            Bucket = self.outputs_bucket,
            Key = path,
            Filename = filename
        )
        logger.info("File uploaded at: {}".format(path))

    def list_objects(self, bucket, path = ""):
        resp = self.s3.list_objects_v2(Bucket=bucket)
        for obj in resp['Contents']:
            files = obj['Key']
        return files


def main():

    s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

    img = s3_helper.get_img("sample/dog1.jpeg")
    logger.info(img)

    files = s3_helper.list_objects(DATASETS_BUCKET)

    logger.info(files)

    s3_helper.upload_file("test-s3-poc/main.py.log")


s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

s3_helper.get_img("sample/dog1.jpeg", True)


if __name__ == "__main__":
    main()
