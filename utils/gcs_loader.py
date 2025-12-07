from google.cloud import storage
import pandas as pd
import io

def list_gcs_blobs(bucket_name, prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return sorted([b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".csv")])

def load_gcs_blob(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(content), low_memory=False)
