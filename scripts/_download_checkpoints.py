import boto3
import pathlib
import sys

BUCKET = "hugo-thesis-runs"
REGION = "eu-north-1"

bundles = [
    "system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_base",
    "system_iv_full_var_unconstrained_principal_grid7x7x7_nz9_state33",
]

s3 = boto3.client("s3", region_name=REGION)
paginator = s3.get_paginator("list_objects_v2")
outbase = pathlib.Path("saved_runs/checkpoints")
outbase.mkdir(parents=True, exist_ok=True)

for bundle in bundles:
    prefix = f"checkpoints/{bundle}/"
    pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            fname = key.split("/")[-1]
            dest = outbase / bundle / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            size_mb = obj["Size"] / 1e6
            print(f"Downloading {fname}  ({size_mb:.1f} MB) ...", flush=True)
            s3.download_file(BUCKET, key, str(dest))
            print(f"  -> {dest}", flush=True)

print("Done.")
