#!/usr/bin/env python3
"""Turn the original labelled bank-transaction sample into the published, anonymised dataset.

The raw file (transactions_train_public.csv in the old `bankclassifier` repo) is NOT part of this
repository. It contained person-to-person payment lines with real-looking account numbers, phone
numbers, names and free-text memos, so none of those survive here. Rules, in order:

 1. Drop any line that looks personal: payment/transfer wording, BSB+account or phone numbers, names,
    initials, free-text memos (anything with no capital letter, or a memo on an internal transfer).
 2. Drop a few merchant-looking lines that could be a person's name or reveal an affiliation.
 3. Replace employer-identifying salary lines with a generic payroll description.
 4. Drop dates (there are no amounts), de-duplicate, keep a `count` of how often each line appeared.
 5. Drop descriptions whose label was ambiguous (the same text carrying different labels).

Usage: python3 tools/sanitise.py path/to/raw.csv > data/transactions.csv
"""
import csv
import re
import sys
from collections import Counter, defaultdict

PERSONAL = re.compile(
    r"(?i:jane doe)|Transfer to|Osko|Deposit|tfer|Tfer|\+61|\b\d{6}\s+\d{5,}\b|"
    r"RYAN|WARD|Ryan|\bMR |\bMS |\bMISS |alistair|Josh|Tan Le|Priscilla|KARPOWICZ|VANDENSCHR|Tracy|Yao|"
    r"Reimbursement|Joint Acc|Antifa"
)
EXTRA_DROP = {  # could be a person's name, an affiliation, or a free-text memo
    "MOLLY DENE", "IMMANUEL BRIGHTON EAST", "JOANNA ROVAS PTY LTD", "NZ", "MMA", "Indonesia", "spare money for Indonesia",
}
INTERNAL_OK = {"Internal Transfer", "Internal Transfer - Internal Transfer"}


def keep(desc: str) -> bool:
    d = desc.strip()
    if not d or d in EXTRA_DROP:
        return False
    if d.startswith("Internal Transfer") or d.endswith("Internal Transfer"):
        return d in INTERNAL_OK
    if PERSONAL.search(d):
        return False
    if not re.search(r"[A-Z]", d):  # memo-like text
        return False
    return True


def generic(desc: str, label: str) -> str:
    if label == "Income" and re.search(r"PEXA|PROPERTY EXCHANG", desc):
        return "EMPLOYER PTY LTD PAYROLL" if desc.startswith("PEXA") else "EMPLOYER PTY LTD BONUS PAYMENT"
    return desc


def main(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace")))
    labels = defaultdict(Counter)
    for r in rows:
        desc, label = r["x_train"].strip(), r["y_train"].strip()
        if keep(desc):
            labels[generic(desc, label)][label] += 1
    out = []
    for desc, c in labels.items():
        top = c.most_common(2)
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # ambiguous
        out.append((desc, top[0][0], sum(c.values())))
    out.sort(key=lambda t: (t[1], t[0]))
    w = csv.writer(sys.stdout, lineterminator="\n")
    w.writerow(["description", "category", "count"])
    w.writerows(out)


if __name__ == "__main__":
    main(sys.argv[1])
