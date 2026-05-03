"""Create AWS Budget alerts for the thesis EC2 sweeps.

One-time setup. Idempotent: re-running updates the existing budget rather than
creating duplicates. Sends email alerts at 80% and 100% of a configurable
monthly cap (default $100).

Usage:
    python scripts/setup_budget.py --email you@example.com
    python scripts/setup_budget.py --email you@example.com --cap-usd 200
    python scripts/setup_budget.py --delete   # remove the budget

This is the SIMPLE cost-cap layer (email warning only). The HARD cap layer is
the 16-hour self-destruct timer baked into scripts/ec2_userdata.sh.

Required IAM: hugo-cli needs `aws-budgets-actions:*` and `budgets:*`. With
IAMFullAccess attached this is already covered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

BUDGET_NAME = "thesis-ec2-monthly-cap"


def get_account_id() -> str:
    cp = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        sys.exit(f"aws sts get-caller-identity failed: {cp.stderr}")
    return cp.stdout.strip()


def budget_exists(account_id: str) -> bool:
    cp = subprocess.run(
        ["aws", "budgets", "describe-budget",
         "--account-id", account_id,
         "--budget-name", BUDGET_NAME],
        capture_output=True, text=True,
    )
    return cp.returncode == 0


def delete_budget(account_id: str) -> None:
    cp = subprocess.run(
        ["aws", "budgets", "delete-budget",
         "--account-id", account_id,
         "--budget-name", BUDGET_NAME],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        sys.exit(f"delete-budget failed: {cp.stderr}")
    print(f"deleted budget '{BUDGET_NAME}'")


def build_budget_doc(cap_usd: float) -> dict:
    return {
        "BudgetName": BUDGET_NAME,
        "BudgetLimit": {"Amount": f"{cap_usd:.2f}", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
        "CostFilters": {},
        "CostTypes": {
            "IncludeTax": True,
            "IncludeSubscription": True,
            "UseBlended": False,
            "IncludeRefund": False,
            "IncludeCredit": False,
            "IncludeUpfront": True,
            "IncludeRecurring": True,
            "IncludeOtherSubscription": True,
            "IncludeSupport": True,
            "IncludeDiscount": True,
            "UseAmortized": False,
        },
    }


def build_notifications(email: str) -> list[dict]:
    return [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
        },
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
        },
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100,
                "ThresholdType": "PERCENTAGE",
                "NotificationState": "ALARM",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
        },
    ]


def create_budget(account_id: str, cap_usd: float, email: str) -> None:
    budget_doc = build_budget_doc(cap_usd)
    notifications = build_notifications(email)
    cp = subprocess.run(
        ["aws", "budgets", "create-budget",
         "--account-id", account_id,
         "--budget", json.dumps(budget_doc),
         "--notifications-with-subscribers", json.dumps(notifications)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        sys.exit(f"create-budget failed: {cp.stderr}")
    print(f"created budget '{BUDGET_NAME}' = ${cap_usd:.2f}/month")
    print(f"alerts (email -> {email}):")
    print(f"  - 80%  of cap = ${0.80 * cap_usd:.2f} actual spend")
    print(f"  - 100% of cap = ${cap_usd:.2f} actual spend")
    print(f"  - 100% of cap = ${cap_usd:.2f} forecasted")
    print()
    print("AWS will send a confirmation email; you must accept it for alerts "
          "to fire.")


def update_budget(account_id: str, cap_usd: float, email: str) -> None:
    budget_doc = build_budget_doc(cap_usd)
    cp = subprocess.run(
        ["aws", "budgets", "update-budget",
         "--account-id", account_id,
         "--new-budget", json.dumps(budget_doc)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        sys.exit(f"update-budget failed: {cp.stderr}")
    print(f"updated budget '{BUDGET_NAME}' = ${cap_usd:.2f}/month "
          "(notifications unchanged)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--email", help="Email address for budget alerts.")
    ap.add_argument("--cap-usd", type=float, default=100.0,
                    help="Monthly cost cap in USD (default 100).")
    ap.add_argument("--delete", action="store_true",
                    help="Delete the budget instead of creating/updating it.")
    args = ap.parse_args()

    account_id = get_account_id()
    print(f"AWS account: {account_id}")

    if args.delete:
        if not budget_exists(account_id):
            print(f"budget '{BUDGET_NAME}' does not exist; nothing to delete")
            return
        delete_budget(account_id)
        return

    if not args.email:
        sys.exit("--email is required (alerts must go somewhere). Or pass --delete.")

    if budget_exists(account_id):
        print(f"budget '{BUDGET_NAME}' exists; updating cap (notifications "
              "kept from prior setup)")
        update_budget(account_id, args.cap_usd, args.email)
    else:
        create_budget(account_id, args.cap_usd, args.email)


if __name__ == "__main__":
    main()
