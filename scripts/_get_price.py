import boto3, json

ec2 = boto3.client('pricing', region_name='us-east-1')
r = ec2.get_products(
    ServiceCode='AmazonEC2',
    Filters=[
        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': 'c6i.32xlarge'},
        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': 'EU (Stockholm)'},
        {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
        {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
        {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
        {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
    ]
)
for p in r['PriceList']:
    d = json.loads(p)
    terms = d.get('terms', {}).get('OnDemand', {})
    for term in terms.values():
        for dim in term['priceDimensions'].values():
            price = float(dim['pricePerUnit']['USD'])
            if price > 0:
                print(f"On-demand: ${price:.4f}/hr")
                print(f"15h:       ${price*15:.2f}")
