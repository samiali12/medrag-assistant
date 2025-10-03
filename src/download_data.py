import os 

from src.constant import BASE_DIR

import boto3
from botocore import UNSIGNED
from botocore.client import Config



TARGET_DIR = os.path.join(BASE_DIR, "data", "demo")

def download_pmc_docs(
        bucket="pmc-oa-opendata",
        prefix="oa_comm/txt/all",
        target_dir=TARGET_DIR,
        limit=2000
):
    os.makedirs(target_dir, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')

    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".txt"):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(target_dir, filename)
            
            if not os.path.exists(local_path):
                s3.download_file(bucket, key, local_path)
                downloaded += 1

            if downloaded >= limit:
                print(f"✅ Reached limit of {limit} documents.")
                return
        
    print(f"✅ Finished. Total downloaded: {downloaded}")

    return True 
 