import boto3
s3 = boto3.client('s3', region_name='eu-north-1')
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='hugo-thesis-runs', Prefix='saved_runs/system_iv_full_var_unconstrained_principal_grid7x7x7_nz11_ret_v1')
found = False
for page in pages:
    for obj in page.get('Contents', []):
        size_mb = obj['Size'] / 1e6
        ts = str(obj['LastModified'])[:19]
        print(f"{obj['Key']}  {size_mb:.1f} MB  {ts}")
        found = True
if not found:
    print("Not in saved_runs yet.")
