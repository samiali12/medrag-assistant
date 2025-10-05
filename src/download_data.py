import os 

from src.constant import BASE_DIR

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm



TARGET_DIR = os.path.join(BASE_DIR, "data", "pmc")

def download_pmc_docs(
        bucket="pmc-oa-opendata",
        prefix="oa_comm/txt/all",
        target_dir=TARGET_DIR,
        limit=1000
):
    os.makedirs(target_dir, exist_ok=True)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')

    downloaded = 0
    pbar = tqdm(total=limit, desc="Downloading PMC documents")

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
                pbar.update(1)

            if downloaded >= limit:
                pbar.close()
                print(f"✅ Reached limit of {limit} documents.")
                return
        
    print(f"✅ Finished. Total downloaded: {downloaded}")
 