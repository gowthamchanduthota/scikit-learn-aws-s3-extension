import boto3

DATASETS_BUCKET = "se-project-ext-datasets"
OUTPUTS_BUCKET = "se-project-ext-outputs"
s3_1 = boto3.client('s3')

paginator = s3_1.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=DATASETS_BUCKET, Prefix="UTKFace")
obj_list = []

for page in pages:
    for obj in page.get('Contents', []):  # Ensure no KeyError if 'Contents' is missing
        obj_list.append(obj['Key'])

# Print the length of the list of objects
print(len(obj_list))

# Uncomment to print the last 5 items in the list (optional)
# print(obj_list[-5:])
