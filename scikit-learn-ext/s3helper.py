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


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger()


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

    def get_img(self, path, show = False, greyscale = False):
        # print(path)
        filename = path.split("/")[-1]
        self.s3.download_file(
            Bucket = self.datasets_bucket,
            Key = path,
            Filename = filename
        )
        img = image.load_img(filename)
        if greyscale:
            img = image.load_img(filename, color_mode="grayscale")
        if show:
            imgrd = mpimg.imread(filename)
            plt.imshow(imgrd)
            plt.show()
        return img

    # Update this fn to be a decorator call fn
    # def upload_file(self, path):
    #     filename = path.split("/")[-1]
    #     self.s3.upload_file(
    #         Bucket = self.outputs_bucket,
    #         Key = path,
    #         Filename = filename
    #     )
    #     logger.info("File uploaded at: {}".format(path))

    def upload_file(self, file_name, s3_key = None, bucket = None):
        # filename = file.split("/")[-1]
        if not s3_key:
            s3_key = os.path.basename(file_name)

        if not bucket:
            bucket = self.outputs_bucket

        self.s3.upload_file(
            Bucket = bucket,
            Key = s3_key,
            Filename = file_name
        )
        logger.info("File uploaded at: {}".format(s3_key))

    def list_objects(self, bucket, path = ""):
        # resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=path)
        # files = []
        # for obj in resp['Contents']:
        #     if obj['Size']: files.append(obj['Key'])

        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=path)
        files = []
        for page in pages:
            for obj in page['Contents']:
                files.append(obj['Key'])
        return files

    def get_frac(self, frac, path = "", random_seed = 42, download_files = False):
        obj_list = self.list_objects(self.datasets_bucket, path)

        dataset_len = len(obj_list)
        frac_len = (int) (dataset_len * frac)
        random.seed(random_seed)
        rand_list = random.sample(obj_list, frac_len)

        if download_files:
            for it in rand_list:
                if not os.path.exists(os.path.dirname(it)):
                    os.makedirs(os.path.dirname(it))

                self.s3.download_file(
                    Bucket = self.datasets_bucket,
                    Key = it,
                    Filename = it
                )
            logger.info("Files downloaded at {}".format(path))

        return rand_list

def main():

    s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

    # get image from
    img = s3_helper.get_img("sample/dog1.jpeg", show = False)
    # logger.i
    logger.info("hehehe")
    logger.info(img)
    logger.info(type(img))

    # To List files
    # files = s3_helper.list_objects(DATASETS_BUCKET, "UTKFace")
    # logger.info(files)

    # Get a fraction of files from
    # frac_files = s3_helper.get_frac(0.1, path="UTKFace", random_seed=42, download_files=True)

    # logger.info("Printing list: ")
    # logger.info(len(frac_files))
    # logger.info(frac_files)
    # logger.info("Done Printing")

    # s3_helper.upload_file(file_name = "main.py.log", s3_key="test-s3-poc/{}/main.py.log".format(time.strftime("%Y%m%d-%H%M%S")))

# ----Advantage Get files on demand or realtime during execution.
# logging.info("hehehe")
if __name__ == "__main__":
    main()
