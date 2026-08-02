#!/usr/bin/env python3
"""
Verifies which of the flagged "missing currency value" bugs from the
23-trace bug report are actually fixed by the new _RE_CURRENCY_VALUE regex.

Filters out non-bugs before scoring:
  - User-stated budget thresholds / impressions ("under $200", "over $3000")
    that never appear as an assistant-confirmed transactional value.
  - Universal boilerplate ($100 airline policy line) - already excluded
    upstream in the bug report, kept excluded here too.

Everything else (assistant-confirmed prices, refunds, fees, certificate/
gift-card balances, totals) is treated as a real bug that must be
recoverable by extraction.

Run this directly against your repo like:
    python3 verify_bug_fixes.py
It will import the *actual* _RE_CURRENCY_VALUE from your codebase if
found on PYTHONPATH / relative path; otherwise it falls back to an
embedded copy of the patched regex so the script is still runnable
standalone for a sanity check.
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Try to import the real, patched regex from the repo. Fall back to an
#    embedded copy (kept identical to the patch) if the repo isn't importable
#    from here, so this script still works standalone.
# ---------------------------------------------------------------------------
_RE_CURRENCY_VALUE = None
REPO_SRC_CANDIDATES = [
    Path("src"),                       # if run from repo root
    Path("/Users/shaurya/python/dagc_pkg_v3_final/src"),
]
for cand in REPO_SRC_CANDIDATES:
    if cand.exists():
        sys.path.insert(0, str(cand))

try:
    from dagc.value_recovery_ext import _RE_CURRENCY_VALUE as _IMPORTED
    _RE_CURRENCY_VALUE = _IMPORTED
    SOURCE = "imported from dagc.value_recovery_ext (real repo regex)"
except Exception:
    # Fallback: embedded copy of the patched regex (must match the patch
    # applied in value_recovery_ext.py). If this drifts from the real repo
    # regex, treat results from this script as advisory only.
    _RE_CURRENCY_VALUE = re.compile(
        r'(?:\$|€|£|¥|₹|₩|₽|₺|R\$|C\$|A\$)\s?(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})?'
        r'|(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})?\s?(?:USD|EUR|GBP|JPY|INR|CNY|AUD|CAD)\b'
    )
    SOURCE = "embedded fallback copy (repo import failed — treat as advisory)"


def normalize(value: str) -> str:
    """Normalize a dollar string for comparison: strip $, commas, whitespace."""
    return re.sub(r'[^\d.]', '', value)


# ---------------------------------------------------------------------------
# 2. Trace data: for each trace, the raw text snippets from the bug report,
#    the values the report flagged as missing, and which of those are
#    filler/non-bugs (user-stated thresholds/impressions never confirmed
#    as an actual transactional value by the assistant) rather than real
#    extraction bugs.
# ---------------------------------------------------------------------------
TRACES = {
    "tau_retail_0015": {
        "text": (
            "The laptop model with item ID 9844888101 is available. "
            "Here are its specifications: Price: $2459.74 "
            "I found the wristwatch with a leather strap and a white dial in your order. "
            "Price Difference: The new model is priced at $1908.15 and your current model is priced at $1985.30"
        ),
        "missing": ["$1985.30", "$2459.74"],
        "filler": [],
    },
    "tau_airline_0067": {
        "text": (
            "JG7FMM: $6,594 refund 2FBBAH: $3,925 refund X7BYG1: $5,418 refund "
            "EQ1G6C: $2,452 refund BOH180: $5,164 refund Total savings: $23,553"
        ),
        "missing": ["$3,925"],
        "filler": [],
    },
    "tau_retail_0007": {
        "text": (
            "Your order #W5061109 currently includes the following items: "
            "Bluetooth Speaker - $302.67 If you need any further assistance "
            "The total price of your order is $1,319.43."
        ),
        "missing": ["$1,319.43", "$302.67"],
        "filler": [],
    },
    "tau_retail_0001": {
        "text": (
            "Could you check if it's under $200? "
            "Can you look it up and see if there's a mechanical keyboard option under $200 for me? "
            "none of the available variants of the mechanical keyboard are priced under $200. "
            "The lowest priced option currently available is $226.11 which features a tactile switch type "
            "Since it's not under $200"
        ),
        "missing": ["$200"],
        "filler": ["$200"],  # user's own budget threshold, never confirmed as a transaction value
    },
    "tau_airline_0069": {
        "text": (
            "You can use: Certificate worth $250 (ID: certificate_6730850) "
            "Certificate worth $100 (ID: certificate_9982533) Once you provide this information and confirm"
        ),
        "missing": ["$250"],
        "filler": [],
    },
    "tau_retail_0041": {
        "text": (
            "The refund of $242.15 will be added to your gift card balance immediately. "
            "New watch: Leather strap, black dial - $1,908.15 Please confirm with \"yes\" "
            "Please confirm if you want me to change your watch from the current white dial to black dial "
            "(keeping the leather strap). This will result in a refund of $77.15 to your gift card. "
            "The refund of $77.15 will be added to your gift card balance immediately."
        ),
        "missing": ["$1,908.15", "$242.15", "$77.15"],
        "filler": [],
    },
    "tau_airline_0033": {
        "text": (
            "Flight HAT069 Departure: 06:00 AM EST Arrival: 12:00 PM EST Price: $121 "
            "Flight HAT083 Departure: 01:00 AM EST Arrival: 07:00 AM EST Price: $100 "
            "The total price for the selected flights is $305. Mia wants to use her $250 certificate "
            "and pay the remaining $55 with her credit card ending in 7447. "
            "Payment: $250 from certificate, $55 from credit card ending in 7447"
        ),
        "missing": ["$121", "$250", "$305", "$55"],
        "filler": [],
    },
    "tau_retail_0058": {
        "text": (
            "Current item: Woody Men's Perfume 30ml (Item ID: 5081446110) at $322.52 "
            "New item: Woody Men's Perfume 100ml (Item ID: 3399869890) at $312.04 "
            "Price difference: $10.48 (to be refunded to your PayPal account)"
        ),
        "missing": ["$312.04", "$322.52"],
        "filler": [],
    },
    "tau_retail_0051": {
        "text": (
            "A delivered order (#W2890441) that includes a water bottle ($54.04) and skateboard "
            "Pet Bed (Item ID: 2751999929) currently $195.11 to the large memory foam beige bed "
            "(Item ID: 6942241102) at $180.93 Office Chair (Item ID: 8069050545) currently $499.28 "
            "to the blue leather chair with no armrests (Item ID: 4168944673) at $471.82 "
            "in need of immediate assistance with exchanging items for cheaper versions to save money ($41.64 potential savings)"
        ),
        "missing": ["$180.93", "$195.11", "$41.64", "$471.82", "$499.28", "$54.04"],
        "filler": [],
    },
    "tau_retail_0057": {
        "text": (
            "Fleece Jacket (size L half zipper) - $153.18 Would you still like me to proceed "
            "Fleece Jacket - $153.18 Total: $1,003.22 Please confirm with \"yes\""
        ),
        "missing": ["$153.18"],
        "filler": [],
    },
    "tau_retail_0044": {
        "text": (
            "The bluetooth speaker costs $302.67 and has a battery life of 20 hours. "
            "I was under the impression it would be under $300. And do you have any bluetooth speakers "
            "available for less than $100? all our bluetooth speakers are priced between $271.89 and $326.59. "
            "From: Blue bluetooth speaker ... at $302.67 To: Blue bluetooth speaker ... at $271.89"
        ),
        "missing": ["$300", "$302.67", "$326.59"],
        "filler": ["$300"],  # user's own impression/budget statement, not a confirmed transaction value
    },
    "tau_retail_0008": {
        "text": (
            "8 hours battery life IPX4 water resistance** - $226.49 Your current earbuds are priced at $256.67. "
            "I'll go with the Blue 8 hours battery life IPX4 water resistance option for $226.49. "
            "New Item: Wireless Earbuds 8 hours battery life IPX4 water resistance - $226.49 "
            "Price Difference: $30.18 (to be refunded to your PayPal account)"
        ),
        "missing": ["$226.49"],
        "filler": [],
    },
    "tau_retail_0010": {
        "text": (
            "I need to confirm the details: Price Difference: The new puzzle costs $46.13 "
            "and the original was $54.11."
        ),
        "missing": ["$54.11"],
        "filler": [],
    },
    "tau_airline_0080": {
        "text": (
            "Gift Cards: #8190333 ($280 available) or #6490722 ($127 available) "
            "Which payment method would you like to use for the cabin upgrade?"
        ),
        "missing": ["$127", "$280"],
        "filler": [],
    },
    "tau_retail_0055": {
        "text": (
            "There's a red helmet in size M with high ventilation available (item ID: 8573379326) for $196.73. "
            "Low ventilation (Item ID: 3358616356) - $197.33 "
            "I found a 2-piece black softshell luggage set (item ID: 8926329222) available for $452.28. "
            "Current set: 3-piece blue softshell (Item ID: 6301799585) - $495.87 "
            "A pending order (#W3561391) for a portable gas grill with no special features ($1,057.24) "
            "A delivered order (#W6876713) for a medium charcoal grill with rotisserie feature ($962.69) "
            "no features (Item ID: 5946177616) - $1,057.24 Want to change to: Charcoal grill with rotisserie "
            "(Item ID: 7082455361) - $962.69 "
            "You'll receive a $0.60 refund ... You'll receive a $43.59 refund ... "
            "You'll receive a $94.55 refund"
        ),
        "missing": ["$0.60", "$1,057.24", "$196.73", "$197.33", "$43.59", "$452.28",
                    "$495.87", "$94.55", "$962.69"],
        "filler": [],
    },
    "tau_retail_0048": {
        "text": (
            "This will reduce your total by $547.55 which will be refunded to your original payment method "
            "(credit card ending in 7455506). I see you have a gift card (gift_card_9368765) with a balance of $85."
        ),
        "missing": ["$547.55", "$85"],
        "filler": [],
    },
    "tau_airline_0032": {
        "text": (
            "New Total Price: $241 (previously $3,538 per passenger) Refund: $6,594 has been processed "
            "back to your credit card ending in 7803. New Total Price: $465 (previously $2,210 per passenger) "
            "Refund: $3,925 has been processed back to your gift card."
        ),
        "missing": ["$2,210", "$3,538"],
        "filler": [],
    },
    "tau_airline_0076": {
        "text": (
            "I have accessed your reservation (3RK2T9) details directly there is only one payment of $280 "
            "made via gift card (gift_card_2550356)"
        ),
        "missing": ["$280"],
        "filler": [],
    },
    "tau_airline_0035": {
        "text": (
            "you are allowed 2 free checked bags per passenger in the economy cabin. "
            "you are allowed a total of 4 free checked bags for your flight. "
            "each additional bag will incur a fee of $50."
        ),
        "missing": ["$50"],
        "filler": [],
    },
    "tau_retail_0042": {
        "text": "Since the price is over $3000",
        "missing": ["$3000"],
        "filler": ["$3000"],  # user's own stated threshold, never a confirmed value
    },
    "tau_retail_0049": {
        "text": (
            "I just placed an order but my card only has $950 limit left. "
            "Total order amount: $1,166.98 Since you mentioned your card has a $950 limit "
            "you have a few options: Remove the Action Camera ... please remove the Action Camera "
            "so total is under $950. The refund of $1,166.98 will be processed to your credit card "
            "ending in 6593 within 5-7 business days. which should bring your total below the $950 limit."
        ),
        "missing": ["$950"],
        "filler": [],  # kept as real: assistant repeats/relies on it as a hard constraint driving the decision
    },
    "tau_retail_0043": {
        "text": (
            "the most budget-friendly mechanical keyboard currently available is the linear switch full size "
            "model at $231.37 (item ID: 9570044148). Linear switch full size - $231.37 Linear switch 60% size - "
            "$232.70 Tactile switch 60% size - $233.90 Linear switch 80% size - $230.15 Tactile switch full size "
            "- $226.11 they're all over $200... A Mechanical Keyboard (60% size RGB backlight) - $236.51"
        ),
        "missing": ["$200", "$226.11", "$230.15", "$231.37", "$232.70", "$233.90", "$236.51"],
        "filler": ["$200"],  # user's own threshold remark, not a confirmed transaction value
    },
    "tau_airline_0074": {
        "text": (
            "BOS to MCO: $122 per person MCO to MSP: $127 per person Total paid for economy: $249 per person "
            "($498 total for both passengers) ... Total for business: $845 per person ($1,690 total for both passengers) "
            "Would you like to proceed with the upgrade to business class for $1,192 and add the 2 checked bags"
        ),
        "missing": ["$1,192", "$498"],
        "filler": [],
    },
}


def main():
    print(f"Regex source: {SOURCE}\n")

    total_real_bugs = 0
    total_fixed = 0
    total_still_broken = 0
    still_broken_report = []

    for trace_id, data in TRACES.items():
        found = {normalize(m.group()) for m in _RE_CURRENCY_VALUE.finditer(data["text"])}
        real_missing = [v for v in data["missing"] if v not in data["filler"]]
        excluded = [v for v in data["missing"] if v in data["filler"]]

        fixed_here = []
        broken_here = []
        for v in real_missing:
            if normalize(v) in found:
                fixed_here.append(v)
            else:
                broken_here.append(v)

        total_real_bugs += len(real_missing)
        total_fixed += len(fixed_here)
        total_still_broken += len(broken_here)

        status = "✅ ALL FIXED" if not broken_here and real_missing else (
            "— no real bugs (all filler)" if not real_missing else "❌ STILL BROKEN"
        )
        print(f"{trace_id}: {status}")
        if fixed_here:
            print(f"  fixed:   {fixed_here}")
        if broken_here:
            print(f"  missing: {broken_here}")
            still_broken_report.append((trace_id, broken_here))
        if excluded:
            print(f"  excluded as filler/non-bug: {excluded}")
        print()

    print("=" * 60)
    print(f"Real bugs tracked:   {total_real_bugs}")
    print(f"Now fixed:           {total_fixed}")
    print(f"Still broken:        {total_still_broken}")
    if total_real_bugs:
        pct = 100 * total_fixed / total_real_bugs
        print(f"Fix rate:            {pct:.1f}%")
    if still_broken_report:
        print("\nTraces still needing work:")
        for tid, vals in still_broken_report:
            print(f"  - {tid}: {vals}")


if __name__ == "__main__":
    main()