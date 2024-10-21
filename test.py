import boto3
import pandas as pd
# from sklearn.model_selection import train_test_spli

# df = pd.read_csv('Advertising.csv')
# print(df.shape)
DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"
s3 = boto3.client('s3')

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=DATASETS_BUCKET, Prefix="UTKFace")
obj_list = []
for page in pages:
    for obj in page['Contents']:
        obj_list.append(obj['Key'])

print(len(lists))
# print(lists[-10:])
