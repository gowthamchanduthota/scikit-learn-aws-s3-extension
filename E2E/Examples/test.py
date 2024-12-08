<<<<<<< HEAD
import base64

def encode(data):
    encoded_bytes = base64.b64encode(data.encode('utf-8'))
    encoded_str = encoded_bytes.decode('utf-8')
    return encoded_str

def decode(data):
    decoded_bytes = base64.b64decode(data)
    decoded_string = decoded_bytes.decode('utf-8')
    return decoded_string
=======
from ..library import DataGenerator, ModelWrapper, s3helper


>>>>>>> 4c0f327 (Added E2E module merging complete changes and modules)
