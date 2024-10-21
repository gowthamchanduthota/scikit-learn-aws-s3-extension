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
        resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=path)
        files = []
        for obj in resp['Contents']:
            if obj['Size']: files.append(obj['Key'])
        return files

    def get_frac(self, frac, path = "", random_state = 42, download_files = False):
        obj_list = self.list_objects(self.DATASETS_BUCKET, path)

        dataset_len = len(obj_list)
        rand_list = random.sample(obj_list, dataset_len * frac)
        if download_files:
            for it in rand_list:
                self.s3.download_file(
                    Bucket = self.DATASETS_BUCKET,
                    Key = it,
                    Filename = it
                )
            logger.info("Files downloaded at {}".format(path))

        return rand_list

def main():

    s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

    # img = s3_helper.get_img("sample/dog1.jpeg")
    # logger.info(img)
    path = "/Users/gowtham/WorkSpace/UNT/SE/datasets/UTKFace"
    dir_list = os.listdir(path)
    dataset_len = len(dir_list)
    print(dataset_len)
    fracc = (int) (dataset_len * 0.1)
    print(fracc)
    rand_lists = random.sample(dir_list, fracc)
    print(len(rand_lists))
    # print(rand_lists)

    # dont undo this.
    # for it in rand_lists:
    #     s3_helper.upload_file("UTKFace/{}".format(it), "UTKFace/{}".format(it), bucket=DATASETS_BUCKET)

    files = s3_helper.list_objects(DATASETS_BUCKET)
    # logger.info("Printing list: ")
    logger.info(len(files))
    # logger.info("Done Printing")

    # s3_helper.upload_file("test-s3-poc/main.py.log")


# s3_helper = S3Helper(DATASETS_BUCKET, OUTPUTS_BUCKET)

# s3_helper.get_img("sample/dog1.jpeg", True)


if __name__ == "__main__":
    main()
