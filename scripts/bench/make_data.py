"""Regenerate scripts/bench/data/*.json (deterministic). The JSON files are the source of truth
for runs; this script documents how they were made. Usage: python scripts/bench/make_data.py"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"

# (string, is_valid, hint)
EMAILS = [
    ("alice@example.com", True, "simple"),
    ("Bob.Smith@Example.org", True, "uppercase letters and non-.com TLD are fine"),
    ("first.last+tag@sub.domain.co.uk", True, "plus tags and multi-label domains are valid"),
    ("user_name@my-company.io", True, "underscore in local part, hyphen inside a domain label"),
    ("x@y.co", True, "one-char local and domain labels are valid"),
    ("a1b2c3@test123.net", True, "digits allowed in local part and labels"),
    ("john%doe@mail.example.com", True, "% is allowed in the local part"),
    ("UPPER@CASE.COM", True, "uppercase TLD is valid"),
    ("dev-ops@ci-cd.dev", True, "hyphens inside labels are valid"),
    ("user+filter+more@gmail.com", True, "multiple + allowed"),
    ("n.a.m.e@domain.travel", True, "single dots between chars are fine; long TLD"),
    ("info@xn--bcher-kva.ch", True, "punycode label: hyphens inside a label are fine"),
    ("_leading.underscore@ok.org", True, "local part may start with underscore"),
    ("100%@percent.net", True, "% allowed, local part may be digits"),
    ("k@a-b.c-d.museum", True, "hyphenated labels, long TLD"),
    ("mixed.Case_99@Sub-Domain.Example.ORG", True, "mixed case everywhere"),
    ("a@b.cd", True, "minimal valid address"),
    ("plus+@plus.com", True, "local part may end with +"),
    ("very.long.local.part.with.many.dots@example.com", True, "many single dots are fine"),
    ("sales-team@123domain.com", True, "label may start with a digit"),
    ("contact@a.b.c.d.e.fr", True, "many labels"),
    ("z_z@zz.zz", True, "two-letter TLD"),
    ("user-@dash.com", True, "local part may end with hyphen (only DOMAIN labels can't)"),
    ("q@localhost.localdomain", True, "two labels, letters-only TLD"),
    ("abc.def@mail-server01.example.net", True, "digits+hyphen inside label"),
    ("ti%@x.io", True, "% at end of local part"),
    ("jane.doe@company.solutions", True, "long TLD"),
    ("o@o.oo", True, "short everything"),
    ("test.email+alex@leetcode.com", True, "dot and plus in local part"),
    ("admin@mailserver1.co.in", True, "label ends with digit"),
    ("A-B_C.D%E+F@g-h.ij", True, "every allowed local-part symbol"),
    ("-dash-first@example.com", True, "local part may start with hyphen"),
    ("a" * 64 + "@example.com", True, "local part of exactly 64 chars is allowed"),
    ("x@" + "d" * 63 + ".com", True, "domain label of exactly 63 chars is allowed"),
    ("Name.Surname@Uni-Stuttgart.DE", True, "uppercase country TLD"),
    ("plainaddress", False, "missing @"),
    ("@missinglocal.com", False, "empty local part"),
    ("missingdomain@", False, "empty domain"),
    ("two@@signs.com", False, "two @ signs"),
    (".leadingdot@example.com", False, "local part cannot start with a dot"),
    ("trailingdot.@example.com", False, "local part cannot end with a dot"),
    ("double..dot@example.com", False, "consecutive dots in local part"),
    ("user@-startdash.com", False, "domain label cannot start with hyphen"),
    ("user@enddash-.com", False, "domain label cannot end with hyphen"),
    ("user@domain..com", False, "empty domain label (consecutive dots)"),
    ("user@domain.com.", False, "trailing dot after domain"),
    ("user@domain", False, "domain needs at least two labels"),
    ("user@domain.c", False, "TLD must be at least 2 letters"),
    ("user@domain.c0m", False, "TLD must be letters only"),
    ("user name@example.com", False, "whitespace not allowed"),
    ("o'reilly@example.com", False, "apostrophe is not in the allowed set"),
    ("user@domain_underscore.com", False, "underscore not allowed in domain"),
    ("user@123.456.789.0", False, "numeric TLD / IP-like domain"),
    ('"quoted"@example.com', False, "quotes not allowed"),
    ("user@[192.168.0.1]", False, "IP-literal domain not allowed"),
    (" user@domain.com", False, "leading whitespace"),
    ("user#hash@example.com", False, "# not allowed"),
    ("<user@example.com>", False, "angle brackets not allowed"),
    ("a" * 65 + "@example.com", False, "local part longer than 64 chars"),
    ("x@" + "d" * 64 + ".com", False, "domain label longer than 63 chars"),
]

DATES = [
    ("2024-03-05", "2024-03-05"), ("1999-12-31", "1999-12-31"), ("2000-02-29", "2000-02-29"),
    ("2024-02-29", "2024-02-29"), (" 2021-07-04 ", "2021-07-04"), ("2024-03-05T10:30:00", "2024-03-05"),
    ("2010-1-9", "2010-01-09"), ("2022/11/30", "2022-11-30"), ("2019.08.15", "2019-08-15"),
    ("2023-02-29", "INVALID"), ("1900-02-29", "INVALID"), ("2021-04-31", "INVALID"),
    ("2021-13-01", "INVALID"), ("2021-00-10", "INVALID"), ("2021-06-00", "INVALID"),
    ("2024-02-30", "INVALID"), ("0999-01-01", "INVALID"),
    ("05/03/2024", "2024-03-05"), ("31/12/1999", "1999-12-31"), ("29/02/2023", "INVALID"),
    ("12/31/2024", "INVALID"), ("1/2/2020", "2020-02-01"), ("15.08.1947", "1947-08-15"),
    ("07-10-2018", "2018-10-07"), ("29/02/2000", "2000-02-29"), ("32/01/2020", "INVALID"),
    ("March 5, 2024", "2024-03-05"), ("Mar 5 2024", "2024-03-05"), ("5 March 2024", "2024-03-05"),
    ("5th March 2024", "2024-03-05"), ("March 5th, 2024", "2024-03-05"), ("1st Jan 2000", "2000-01-01"),
    ("Sept 22, 2021", "2021-09-22"), ("february 29, 2024", "2024-02-29"), ("February 29, 2023", "INVALID"),
    ("DECEMBER 25, 2020", "2020-12-25"), ("23rd of October 2015", "2015-10-23"),
    ("Tuesday, March 5, 2024", "2024-03-05"), ("June 31, 2021", "INVALID"), ("Jun. 3, 2019", "2019-06-03"),
    ("2 May 1990", "1990-05-02"), ("Oct 09, 2023", "2023-10-09"), ("  aug 1 1985", "1985-08-01"),
    ("Febtember 3, 2020", "INVALID"), ("Nov 31 2022", "INVALID"), ("22nd Feb 1900", "1900-02-22"),
    ("20240305", "2024-03-05"), ("19991301", "INVALID"), ("20000229", "2000-02-29"),
    ("05/03/24", "INVALID"), ("Mar 5 '24", "INVALID"), ("", "INVALID"), ("not a date", "INVALID"),
    ("2024", "INVALID"), ("March 2024", "INVALID"), ("2024-03-05-01", "INVALID"),
    ("Fri, 13 Sep 2019", "2019-09-13"), ("2016/2/29", "2016-02-29"), ("31.04.2021", "INVALID"),
    ("Thursday 1 January 1970", "1970-01-01"),
]

# (ticket text, queue, priority)
TICKETS = [
    ("Subject: Refund request | I cancelled within the trial but still see the subscription fee. Please refund.", "billing", "P3"),
    ("Subject: Charged twice | My credit card was charged twice for the March invoice.", "billing", "P2"),
    ("Subject: Invoice copy | Can you send a PDF receipt for my last payment? Need it for taxes.", "billing", "P3"),
    ("Subject: Overcharged | I was overcharged by $40 compared to my plan price.", "billing", "P2"),
    ("Subject: Payment failed | Payment failed when I tried to upgrade, card is valid.", "billing", "P3"),
    ("Subject: Fraud?? | There are three purchases on my card I never made. Possible fraud on my account.", "billing", "P1"),
    ("Subject: Double charged after password reset | I reset my password and then got double charged for the month.", "billing", "P2"),
    ("Subject: Billed wrong amount | I was billed for 10 seats but we only have 6.", "billing", "P3"),
    ("Subject: URGENT refund | Urgent: I need the refund for the duplicate order processed today.", "billing", "P2"),
    ("Subject: Annual invoice | Please change the company name on the annual invoice.", "billing", "P3"),
    ("Subject: Card charged after cancel | I deleted my account last week but my card was charged again.", "billing", "P3"),
    ("Subject: Tax on receipt | The receipt is missing our VAT number.", "billing", "P3"),
    ("Subject: Cannot log in | I can't log in since this morning, it says invalid credentials.", "account", "P3"),
    ("Subject: Locked out | I'm locked out after too many attempts and have a demo in an hour.", "account", "P2"),
    ("Subject: 2FA codes | My 2FA codes stopped working after I changed phones.", "account", "P3"),
    ("Subject: Change username | How do I change my username?", "account", "P3"),
    ("Subject: Password reset email | The password reset email never shows up.", "account", "P3"),
    ("Subject: Account hacked | I think my account was hacked, someone changed my email address on my account.", "account", "P1"),
    ("Subject: Delete my account | Please delete my account and all my data per GDPR.", "account", "P3"),
    ("Subject: Two-factor | Can I turn off two-factor for my team members?", "account", "P3"),
    ("Subject: Sign in loop | Every time I sign in it sends me back to the sign in page.", "account", "P3"),
    ("Subject: locked out of admin | Our only admin is locked out, urgent, nobody can manage users.", "account", "P2"),
    ("Subject: SSO login | Login via Google SSO shows a blank page for me.", "account", "P3"),
    ("Subject: Security alert | Got a security alert about a login from another country. Was that you?", "account", "P1"),
    ("Subject: Where is my package | My package was supposed to arrive Monday, tracking hasn't updated.", "shipping", "P3"),
    ("Subject: Never arrived | Order #5521 never arrived, courier says delivered.", "shipping", "P2"),
    ("Subject: Damaged | The box arrived damaged and the device screen is cracked.", "shipping", "P3"),
    ("Subject: Change shipping address | Can I change the shipping address on an order that hasn't shipped?", "shipping", "P3"),
    ("Subject: Delivery time | How long does delivery to Canada take?", "shipping", "P3"),
    ("Subject: Wrong item shipped | You shipped me the blue model instead of the black one.", "shipping", "P3"),
    ("Subject: Tracking number | I never got a tracking number for my order.", "shipping", "P3"),
    ("Subject: Urgent delivery | Urgent: our package with event badges must arrive by Friday.", "shipping", "P2"),
    ("Subject: Courier lost it | The courier lost my parcel, it never arrived. Need a replacement.", "shipping", "P2"),
    ("Subject: Outage | Your dashboard is down for all users in our org, total outage since 9am.", "technical", "P1"),
    ("Subject: API 500 errors | The API returns 500 errors on every POST to /orders.", "technical", "P3"),
    ("Subject: App crash | The iOS app crashes when I open settings.", "technical", "P3"),
    ("Subject: Data loss after sync | After the last sync half of my notes are gone. This is data loss!", "technical", "P1"),
    ("Subject: Webhook timeout | Webhooks to our endpoint time out with a timeout error after 10s.", "technical", "P3"),
    ("Subject: Page not loading | The reports page is not loading in Firefox.", "technical", "P3"),
    ("Subject: Bug in export | Found a bug: CSV export drops the last row.", "technical", "P3"),
    ("Subject: Production is down | Production is down, our checkout depends on your API!", "technical", "P1"),
    ("Subject: Slow sync | Sync takes 20 minutes now instead of 1.", "technical", "P3"),
    ("Subject: Error on upload | Uploading a 2MB file gives 'unknown error'.", "technical", "P3"),
    ("Subject: Urgent bug | Urgent: a bug in the scheduler sends reminders twice to customers.", "technical", "P2"),
    ("Subject: Security vulnerability | I found an XSS security hole in your comment widget.", "technical", "P1"),
    ("Subject: Integration down | The Slack integration is down for us.", "technical", "P3"),
    ("Subject: Feature request | Would love a dark mode!", "general", "P3"),
    ("Subject: Partnership | We'd like to discuss a reseller partnership.", "general", "P3"),
    ("Subject: Thanks | Just wanted to say the new update is great.", "general", "P3"),
    ("Subject: Office hours | What are your support hours on weekends?", "general", "P3"),
    ("Subject: Press inquiry | I'm a journalist writing about productivity tools.", "general", "P3"),
    ("Subject: Urgent question | Urgent question: do you offer discounts for nonprofits?", "general", "P3"),
    ("Subject: Job application | Are you hiring designers?", "general", "P3"),
    ("Subject: Roadmap | Is there an Android tablet version planned?", "general", "P3"),
    ("Subject: Docs typo | There's a typo on your pricing page FAQ.", "general", "P3"),
    ("Subject: Conference | Will you be at the SaaS summit in May?", "general", "P3"),
    ("Subject: Refund for outage | Your outage last week cost us a day, we want a refund for that month.", "billing", "P1"),
    ("Subject: Login error | Login fails with error code 403 since the update.", "account", "P3"),
    ("Subject: Hacked & charged | My account was hacked and someone charged my card twice.", "billing", "P1"),
    ("Subject: Package and invoice | The package arrived fine but I need an invoice with my address.", "billing", "P3"),
]


def assign_splits(items: list, strata) -> dict[str, list]:
    """Stratified, deterministic 24/18/18 split (pattern T T V S T V S T V S over shuffled strata)."""
    rng = random.Random(1234)
    groups: dict = {}
    for it in items:
        groups.setdefault(strata(it), []).append(it)
    ordered = []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        ordered.extend(g)
    pattern = ["train", "train", "val", "test", "train", "val", "test", "train", "val", "test"]
    out = {"train": [], "val": [], "test": []}
    for i, it in enumerate(ordered):
        out[pattern[i % 10]].append(it)
    return out


def build(task: str, rows: list[dict], strata) -> None:
    assert len(rows) == 60, (task, len(rows))
    splits = assign_splits(rows, strata)
    for name, items in splits.items():
        for ex in items:
            ex["split"] = name
    (OUT / f"{task}.json").write_text(json.dumps(splits, indent=1) + "\n")
    print(task, {k: len(v) for k, v in splits.items()})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    emails = [dict(id=f"em{i:02d}", input=s, label=v, hint=h) for i, (s, v, h) in enumerate(EMAILS)]
    build("regex_emails", emails, lambda e: e["label"])
    dates = [dict(id=f"dt{i:02d}", input=s, expected=e) for i, (s, e) in enumerate(DATES)]
    build("parse_dates", dates, lambda e: e["expected"] == "INVALID")
    tickets = [dict(id=f"tk{i:02d}", input=t, queue=q, priority=p) for i, (t, q, p) in enumerate(TICKETS)]
    build("ticket_router", tickets, lambda e: e["queue"])


if __name__ == "__main__":
    main()
