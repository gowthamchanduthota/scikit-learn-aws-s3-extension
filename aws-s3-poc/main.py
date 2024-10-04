import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import boto3
from tensorflow.keras.preprocessing import image
## POC only
# boto3 AWS S3 Client, connect to s3 client and functions for pull and push

DATASETS_BUCKET = "mohithdataset"
OUTPUTS_BUCKET = "mohithoutput"

s3 = boto3.resource('s3')
bucket = s3.Bucket(DATASETS_BUCKET)


class S3Helper:
    def __init__(self, datasets_bucket, outputs_bucket, credentials = None):
        if credentials:
            #TODO
            print("Using Credentials: ", credentials)
            # update boto client to use credentials
        self.s3_client = boto3.resource('s3')
        self.datasets_bucket = datasets_bucket
        self.outputs_bucket = outputs_bucket

      

    def list_objects(bucket, path = ""):
        for file_item in bucket.objects.all():
            print(file_item.key)
        # use path





## POC only
# boto3 AWS S3 Client, connect to s3 client and functions for pull and push
