import boto3

# Check eu-north-1 for any HPC instance availability
ec2 = boto3.client('ec2', region_name='eu-north-1')
r = ec2.describe_instance_type_offerings(
    LocationType='region',
    Filters=[{'Name': 'instance-type', 'Values': ['hpc*']}]
)
offerings = sorted([t['InstanceType'] for t in r['InstanceTypeOfferings']])
print(f"eu-north-1 HPC offerings: {offerings if offerings else 'NONE'}")
print()

# Check other major regions for hpc8a specifically
print("hpc8a availability across regions:")
for region in ['us-east-1', 'us-east-2', 'us-west-2', 'eu-west-1', 'eu-central-1', 'eu-north-1']:
    rc = boto3.client('ec2', region_name=region)
    try:
        rr = rc.describe_instance_type_offerings(
            LocationType='region',
            Filters=[{'Name': 'instance-type', 'Values': ['hpc8a*']}]
        )
        avail = sorted([t['InstanceType'] for t in rr['InstanceTypeOfferings']])
        print(f"  {region:15s}: {avail if avail else 'NONE'}")
    except Exception as e:
        print(f"  {region:15s}: error: {e}")

# Check HPC vCPU quota in eu-north-1
print()
sq = boto3.client('service-quotas', region_name='eu-north-1')
# Quota code for HPC instance vCPU is L-F7808C92 (Running On-Demand HPC instances)
try:
    q = sq.get_service_quota(ServiceCode='ec2', QuotaCode='L-F7808C92')
    print(f"HPC On-Demand vCPU quota (eu-north-1): {q['Quota']['Value']}")
except Exception as e:
    print(f"HPC quota lookup error: {e}")
