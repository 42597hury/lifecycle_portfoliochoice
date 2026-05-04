import boto3

ce = boto3.client('ce', region_name='us-east-1')
r = ce.get_cost_and_usage(
    TimePeriod={'Start': '2026-05-01', 'End': '2026-05-04'},
    Granularity='DAILY',
    Metrics=['UnblendedCost'],
    GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
)
total = 0.0
for day in r['ResultsByTime']:
    date = day['TimePeriod']['Start']
    print(f"--- {date} ---")
    for g in sorted(day['Groups'], key=lambda x: float(x['Metrics']['UnblendedCost']['Amount']), reverse=True):
        amt = float(g['Metrics']['UnblendedCost']['Amount'])
        if amt > 0.0001:
            svc = g['Keys'][0]
            print(f"  {svc:45s}  ${amt:.4f}")
            total += amt

print(f"\nTotal May 1-3: ${total:.2f}")
