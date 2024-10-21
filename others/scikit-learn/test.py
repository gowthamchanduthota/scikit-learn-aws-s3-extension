import boto3
import pandas as pd
# from sklearn.model_selection import
from sklearn.model_selection import train_test_split

df = pd.read_csv('Advertising.csv')
print(df.shape)