import boto3
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from src.logger import logging
from io import StringIO

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class s3_operations:
    def __init__(self, bucket_name="mlopsnlp26", aws_access_key=None, aws_secret_key=None, region_name="ap-southeast-2"):
        """
        Initialize the s3_operations class with AWS credentials and S3 bucket details.
        If credentials are not provided, they are loaded automatically from the .env file.
        """
        self.bucket_name = bucket_name

        # Retrieve keys from parameters or fallback to .env / OS environment variables
        aws_access_key = aws_access_key or os.getenv("secret-key") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = aws_secret_key or os.getenv("access-key") or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key or not aws_secret_key:
            logging.warning("AWS credentials (access key and secret key) could not be found in .env or environment variables. S3 operations will be disabled.")
            self.s3_client = None
        else:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region_name
            )
            logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        if self.s3_client is None:
            logging.warning("S3 client is not initialized due to missing credentials. Skipping S3 fetch.")
            return None

        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            return None

# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual AWS credentials and S3 details
#     BUCKET_NAME = "bucket-name"
#     AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
#     AWS_SECRET_KEY = "AWS_SECRET_KEY"
#     FILE_KEY = "data.csv"  # Path inside S3 bucket

#     data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame
