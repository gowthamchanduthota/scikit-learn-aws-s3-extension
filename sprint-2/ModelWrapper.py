from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import boto3
import pickle
import os
from s3helper import *

outputs_bucket = "se-project-ext-outputs"

class S3ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, save_model = False, save_metrics = False,  s3_bucket = outputs_bucket):

        self.base_model = base_model
        self.s3_client = S3Helper(outputs_bucket = s3_bucket)
        # self.s3_client = S3Helper(outputs_bucket = s3_bucket, credentials= {
        #     "AWS_ACCESS_KEY_ID" :  aws_access_key_id,
        #     "AWS_SECRET_ACCESS_KEY": aws_secret_access_key
        # })
        self.s3_bucket = s3_bucket
        self.save_model = save_model
        self.save_metrics = save_metrics

    def fit(self, X, *args, **kwargs):
        self.base_model.fit(X, *args, **kwargs)

        # if self.save_model:
        #     self.save_model_to_s3("modell.pkl")

        return self

    def predict(self, X, *args, **kwargs):
        predictions = self.base_model.predict(X, *args, **kwargs)

        # if self.save_metrics:
        #     self.save_predictions_to_s3(predictions= predictions, filename="modelpred.csv")

        return predictions

    def compile(self, *args, **kwargs):
        complied_model = self.base_model.compile(*args, **kwargs)
        return complied_model

    def summary(self, *args, **kwargs):
        summary = self.base_model.summary(*args, **kwargs)
        return summary


    def save_model_to_s3(self, model_filename):
        # TODO - Sprint #3
        # Save model to S3 outputs buckets.
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.base_model, model_file)


        # self.s3_client.upload_file(model_filename, self.s3_bucket, model_filename)
        # # os.remove(model_filename)
        # print(f"Model saved to S3 bucket '{self.s3_bucket}' as {model_filename}.")

    # TODO - Wrapp with py decorator
    def save_predictions_to_s3(self, predictions, filename):

        # TODO - Sprint #3
        pass
        # pd.DataFrame(predictions).to_csv(filename, index=False)


        # self.s3_client.upload_file(filename, self.s3_bucket, filename)
        # os.remove(filename)
        # print(f"Predictions saved to S3 bucket '{self.s3_bucket}' as {filename}.")


def usage():
    # sample code
    base_model = LogisticRegression()
    model = S3ModelWrapper(base_model, s3_bucket=outputs_bucket)

    model.fit(X_train, y_train)

    predictions = s3_model.predict(X_test)

    s3_model.save_model_to_s3('logistic_model.pkl')
    s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
