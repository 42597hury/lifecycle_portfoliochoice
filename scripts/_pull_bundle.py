import boto3, pathlib, time, sys

BUCKET = "hugo-thesis-runs"
REGION = "eu-north-1"

# default to v4_lobatto checkpoint, or pass a bundle name as first arg
bundle = sys.argv[1] if len(sys.argv) > 1 else "system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_v4_lobatto"
prefix_root = sys.argv[2] if len(sys.argv) > 2 else "checkpoints"  # or "saved_runs"

s3 = boto3.client("s3", region_name=REGION)
paginator = s3.get_paginator("list_objects_v2")
outbase = pathlib.Path("saved_runs") / ("checkpoints/" + bundle if prefix_root == "checkpoints" else bundle)
outbase.mkdir(parents=True, exist_ok=True)

pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{prefix_root}/{bundle}/")
for page in pages:
    for obj in page.get("Contents", []):
        fname = obj["Key"].split("/")[-1]
        dest = outbase / fname
        size_mb = obj["Size"] / 1e6
        for attempt in range(5):
            try:
                print(f"Downloading {fname} ({size_mb:.1f} MB)...", flush=True)
                s3.download_file(BUCKET, obj["Key"], str(dest))
                print(f"  -> {dest}", flush=True)
                break
            except Exception as e:
                print(f"  attempt {attempt+1} failed: {e}, retrying...", flush=True)
                time.sleep(5)

print("Done.")
