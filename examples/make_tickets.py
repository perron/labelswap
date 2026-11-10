"""Write a synthetic, made-up ticket log (examples/tickets.jsonl) for the demo.

Each row pretends an LLM already labeled the ticket. Nothing here is real data.
"""
import json
import random
import sys

TEMPLATES = {
    "billing": ["I was charged twice for my {thing} subscription", "refund request for invoice {n}", "my card was declined when paying for {thing}", "why did my bill go up this month"],
    "outage": ["{thing} is down and returns 503 errors", "the {thing} dashboard will not load since {n} minutes", "alarm fired: {thing} latency over threshold", "users report {thing} timing out"],
    "access": ["please reset my password for {thing}", "I cannot log in to {thing} after the update", "grant me admin access to the {thing} project", "my two factor code for {thing} is not arriving"],
    "feature": ["could you add dark mode to {thing}", "it would be great if {thing} exported to csv", "feature idea: let {thing} schedule reports", "please support keyboard shortcuts in {thing}"],
    "security": ["suspicious login to {thing} from an unknown country", "found an exposed api key in the {thing} repo", "phishing email pretending to be {thing}", "vulnerability scan flagged {thing} with a critical cve"],
}
THINGS = ["checkout", "search", "the mobile app", "analytics", "the api gateway", "email", "storage", "the admin panel"]


def main(n=600, seed=7, out="examples/tickets.jsonl"):
    rng = random.Random(seed)
    labels = list(TEMPLATES)
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n):
            label = rng.choice(labels)
            text = rng.choice(TEMPLATES[label]).format(thing=rng.choice(THINGS), n=rng.randint(10, 9999))
            if rng.random() < 0.15:  # ambiguous tickets mix two topics
                other = rng.choice(labels)
                text += " and also " + rng.choice(TEMPLATES[other]).format(thing=rng.choice(THINGS), n=i)
            f.write(json.dumps({"id": i, "text": text, "label": label}) + "\n")


if __name__ == "__main__":
    main(out=sys.argv[1] if len(sys.argv) > 1 else "examples/tickets.jsonl")
