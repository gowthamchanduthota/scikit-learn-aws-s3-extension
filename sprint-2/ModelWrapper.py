from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import boto3
import pickle
import os
from s3helper import *
<<<<<<< HEAD
import logging

outputs_bucket = "se-project-ext-outputs"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, save_model=False, save_metrics=False, s3_bucket=outputs_bucket):
        self.base_model = base_model
        self.s3_client = S3Helper(outputs_bucket=s3_bucket)
        self.s3_bucket = s3_bucket
        self.save_model = save_model
        self.save_metrics = save_metrics
        self.is_fitted = False  # Track if the model is trained

    def fit(self, X, *args, **kwargs):
        self.base_model.fit(X, *args, **kwargs)
        self.is_fitted = True  # Mark model as fitted
        logger.info("Model fitted successfully.")

        # Save the model to S3 if required
        if self.save_model:
            self.save_model_to_s3("model.pkl")

        return self

    def predict(self, X, *args, **kwargs):
        if not self.is_fitted:
            logger.error("Model is not fitted yet. Please train the model before predicting.")
            raise ValueError("Model is not fitted yet.")
        
        predictions = self.base_model.predict(X, *args, **kwargs)

        # Save predictions to S3 if required
        if self.save_metrics:
            self.save_predictions_to_s3(predictions, "modelpred.csv")

        return predictions

    def compile(self, *args, **kwargs):
        complied_model = self.base_model.compile(*args, **kwargs)
        return complied_model

    def summary(self, *args, **kwargs):
        summary = self.base_model.summary(*args, **kwargs)
        return summary

    def save_model_to_s3(self, model_filename):
        """ Save model to S3 bucket """
        logger.info(f"Saving model to S3 bucket '{self.s3_bucket}' as {model_filename}.")
        try:
            with open(model_filename, 'wb') as model_file:
                pickle.dump(self.base_model, model_file)
            self.s3_client.upload_file(model_filename, self.s3_bucket, model_filename)
            os.remove(model_filename)
            logger.info(f"Model saved to S3 successfully as {model_filename}.")
        except Exception as e:
            logger.error(f"Error saving model to S3: {e}")

    def save_predictions_to_s3(self, predictions, filename):
        """ Save predictions to S3 bucket """
        logger.info(f"Saving predictions to S3 bucket '{self.s3_bucket}' as {filename}.")
        try:
            import pandas as pd  # Importing here to ensure it's not always loaded
            pd.DataFrame(predictions).to_csv(filename, index=False)
            self.s3_client.upload_file(filename, self.s3_bucket, filename)
            os.remove(filename)
            logger.info(f"Predictions saved to S3 successfully as {filename}.")
        except Exception as e:
            logger.error(f"Error saving predictions to S3: {e}")

    def evaluate(self, X_test, y_test):
        """ Evaluate the model on test data """
        if not self.is_fitted:
            logger.error("Model is not fitted yet. Please train the model before evaluating.")
            raise ValueError("Model is not fitted yet.")
        
        score = self.base_model.score(X_test, y_test)
        logger.info(f"Model evaluation score: {score}")
        return score


def usage():
    # sample code
    base_model = LogisticRegression()
    s3_model = S3ModelWrapper(base_model, s3_bucket=outputs_bucket)

    s3_model.fit(X_train, y_train)

    predictions = s3_model.predict(X_test)

    # Save the model and predictions to S3
    # s3_model.save_model_to_s3('logistic_model.pkl')
    # s3_model.save_predictions_to_s3(predictions, 'predictions.csv')

    # Evaluate the model on test data
    score = s3_model.evaluate(X_test, y_test)
    print(f"Model test score: {score}")

=======

outputs_bucket = "se-project-ext-outputs"

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
class S3ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, save_model = False, save_metrics = False,  s3_bucket = outputs_bucket):

        self.base_model = base_model
        self.s3_client = S3Helper(outputs_bucket = s3_bucket)
        # self.s3_client = S3Helper(outputs_bucket = s3_bucket, credentials= {
        #     "AWS_ACCESS_KEY_ID" :  aws_access_key_id,
        #     "AWS_SECRET_ACCESS_KEY": aws_secret_access_key
        # })
=======
=======
>>>>>>> cf67fc8 (Added modelWrapper)
class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, save_model = False, save_metrics = False,  s3_bucket = outputs_bucket, aws_access_key_id=None, aws_secret_access_key=None):

        self.base_model = base_model
        self.s3_client = S3Helper(outputs_bucket = s3_bucket, credentials= {
            "AWS_ACCESS_KEY_ID" :  aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key
        })
<<<<<<< HEAD
>>>>>>> 2bf8c60 (Added modelWrapper)
=======
class S3ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, save_model = False, save_metrics = False,  s3_bucket = outputs_bucket):

        self.base_model = base_model
=======
class S3ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, save_model = False, save_metrics = False,  s3_bucket = outputs_bucket):

        self.base_model = base_model
>>>>>>> 8819a0f (Added testcase file)
        self.s3_client = S3Helper(outputs_bucket = s3_bucket)
        # self.s3_client = S3Helper(outputs_bucket = s3_bucket, credentials= {
        #     "AWS_ACCESS_KEY_ID" :  aws_access_key_id,
        #     "AWS_SECRET_ACCESS_KEY": aws_secret_access_key
        # })
<<<<<<< HEAD
>>>>>>> 3119f38 (Added testcase file)
=======
>>>>>>> cf67fc8 (Added modelWrapper)
=======
>>>>>>> 8819a0f (Added testcase file)
        self.s3_bucket = s3_bucket
        self.save_model = save_model
        self.save_metrics = save_metrics

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def fit(self, X, *args, **kwargs):
        self.base_model.fit(X, *args, **kwargs)
=======
    def fit(self, X, y):
        self.base_model.fit(X, y)
>>>>>>> 2bf8c60 (Added modelWrapper)
=======
    def fit(self, X, *args, **kwargs):
        self.base_model.fit(X, *args, **kwargs)
>>>>>>> 3119f38 (Added testcase file)
=======
    def fit(self, X, y):
        self.base_model.fit(X, y)
>>>>>>> cf67fc8 (Added modelWrapper)
=======
    def fit(self, X, *args, **kwargs):
        self.base_model.fit(X, *args, **kwargs)
>>>>>>> 8819a0f (Added testcase file)

        # if self.save_model:
        #     self.save_model_to_s3("modell.pkl")

        return self

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def predict(self, X, *args, **kwargs):
        predictions = self.base_model.predict(X, *args, **kwargs)
=======
    def predict(self, X):
        predictions = self.base_model.predict(X)
>>>>>>> 2bf8c60 (Added modelWrapper)
=======
    def predict(self, X, *args, **kwargs):
        predictions = self.base_model.predict(X, *args, **kwargs)
>>>>>>> 3119f38 (Added testcase file)
=======
    def predict(self, X):
        predictions = self.base_model.predict(X)
>>>>>>> cf67fc8 (Added modelWrapper)
=======
    def predict(self, X, *args, **kwargs):
        predictions = self.base_model.predict(X, *args, **kwargs)
>>>>>>> 8819a0f (Added testcase file)

        # if self.save_metrics:
        #     self.save_predictions_to_s3(predictions= predictions, filename="modelpred.csv")

        return predictions

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3119f38 (Added testcase file)
=======
>>>>>>> 8819a0f (Added testcase file)
    def compile(self, *args, **kwargs):
        complied_model = self.base_model.compile(*args, **kwargs)
        return complied_model

    def summary(self, *args, **kwargs):
        summary = self.base_model.summary(*args, **kwargs)
        return summary

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 2bf8c60 (Added modelWrapper)
=======
>>>>>>> 3119f38 (Added testcase file)
=======
>>>>>>> cf67fc8 (Added modelWrapper)
=======
>>>>>>> 8819a0f (Added testcase file)

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


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 3119f38 (Added testcase file)
=======
>>>>>>> 8819a0f (Added testcase file)
def usage():
    # sample code
    base_model = LogisticRegression()
    model = S3ModelWrapper(base_model, s3_bucket=outputs_bucket)
<<<<<<< HEAD
<<<<<<< HEAD

    model.fit(X_train, y_train)

    predictions = s3_model.predict(X_test)

<<<<<<< HEAD
# s3_model.save_model_to_s3('logistic_model.pkl')
# s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
>>>>>>> c803f78 (Added modelWrapper)
=======
    s3_model.save_model_to_s3('logistic_model.pkl')
    s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
>>>>>>> a56ddc3 (Added testcase file)
=======
# base_model = LogisticRegression()
# model = S3ModelWrapper(base_model, s3_bucket=outputs_bucket)
=======
>>>>>>> 3119f38 (Added testcase file)

    model.fit(X_train, y_train)

    predictions = s3_model.predict(X_test)

<<<<<<< HEAD
# s3_model.save_model_to_s3('logistic_model.pkl')
# s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
>>>>>>> ed915f3 (Added modelWrapper)
<<<<<<< HEAD
>>>>>>> 2bf8c60 (Added modelWrapper)
=======
=======
    s3_model.save_model_to_s3('logistic_model.pkl')
    s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
>>>>>>> 175d09d (Added testcase file)
>>>>>>> 3119f38 (Added testcase file)
=======
# base_model = LogisticRegression()
# model = S3ModelWrapper(base_model, s3_bucket=outputs_bucket)
=======
>>>>>>> 8819a0f (Added testcase file)

    model.fit(X_train, y_train)

    predictions = s3_model.predict(X_test)

<<<<<<< HEAD
# s3_model.save_model_to_s3('logistic_model.pkl')
# s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
>>>>>>> c803f78 (Added modelWrapper)
<<<<<<< HEAD
>>>>>>> cf67fc8 (Added modelWrapper)
=======
=======
    s3_model.save_model_to_s3('logistic_model.pkl')
    s3_model.save_predictions_to_s3(predictions, 'predictions.csv')
>>>>>>> a56ddc3 (Added testcase file)
>>>>>>> 8819a0f (Added testcase file)
