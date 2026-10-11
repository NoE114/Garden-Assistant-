"""Audit what the model actually says against the Python-computed facts.

Two invariants:
  1. No day-count in the output that is not in the facts handed to the model.
  2. No plant named in the output that was not in the allowed list.

This loads the GGUF and runs real generations, so it is slow (~1-2 min).
Kept out of the fast pure-Python suites on purpose:

    .venv/bin/python test_model_audit.py [region ...]

Every region is audited at the real date, and the full set is audited once more
with the clock pinned to a spring date, so the in-season prompt path gets the
same scrutiny as the off-season one.
"""

import re
import sys

from frost import FROST_DATES, OFFSEASON_TASKS, set_today, status
from garden import advise, deterministic_answer

# Audited in addition to the real current date. April exercises the in-season
# prompt with a full crop list, which October never does.
SPRING_DAY = "2026-04-15"

# The full vocabulary the program knows about. Anything outside this that looks
# like a plant name is a hallucination.
KNOWN_PLANTS = {
    "broccoli", "kale", "peas", "spinach", "lettuce", "tomatoes", "peppers",
    "cucumbers", "basil", "squash", "beets", "carrots", "chard", "cilantro",
    "garlic", "cover", "compost",
}
# Substring stems -> canonical names, so plural/singular forms are tolerated.
PLANT_STEMS = {
    "broc": "broccoli", "kal": "kale", "pea": "peas", "spin": "spinach",
    "lettu": "lettuce", "tomat": "tomatoes", "pepper": "peppers",
    "cucum": "cucumbers", "basil": "basil", "squash": "squash",
    "beet": "beets", "carrot": "carrots", "chard": "chard",
    "cilantro": "cilantro", "garlic": "garlic",
}
# Words that contain a stem but are not plants.
FALSE_FRIENDS = {"spread", "spear", "appeal", "repeat", "pearl", "meal", "heal",
                 "peak", "leap", "cheap", "steal", "real", "ideal", "speak"}


def allowed_numbers(region, override=None):
    """Every number the model was given, plus numbers baked into the task list.

    The off-season prompt contains fixed instructions like "6 inches apart";
    those come from Python, so they are legitimate output too.
    """
    s = status(region, override)
    allowed = set()
    for key in ("frost_free_days", "days_until_fall_frost", "days_until_last_frost"):
        allowed.add(str(s[key]))
    for iso in (s["today"], s["last_spring_frost"], s["first_fall_frost"]):
        for part in iso.replace("-", " ").split():
            allowed.add(part)
    for task in OFFSEASON_TASKS:
        for n in re.findall(r"\d+", task):
            allowed.add(n)
    return allowed


def allowed_plants(region, override=None):
    s = status(region, override)
    return set(s["hardy_now"] + s["root_now"] + s["tender_now"] + s["fall_planted"])


def audit(region, override=None):
    """Return (bad_numbers, bad_plants, text) for one region."""
    text = advise(region, override)
    nums = allowed_numbers(region, override)

    # Dates in the output (2026-10-10) are fine if they match a given date.
    flat = text.replace("-", " ")
    # Strip the numbered-list markers the prompt explicitly asks for
    # ("1. 2. 3."), so they are not mistaken for invented day-counts.
    flat = re.sub(r"(?:^|\s)\d+\s*[.)]\s", " ", flat)
    bad_nums = set()
    for n in re.findall(r"\b\d{1,4}\b", flat):
        if n not in nums:
            bad_nums.add(n)

    # The model going silent is the one failure this audit cannot otherwise
    # see: advise() falls back to deterministic_answer() on empty output, and
    # that fallback is by construction free of invented numbers and plants, so
    # it would pass every check below while no AI ran at all. Assert the text
    # is genuinely the model's.
    is_fallback = text.strip() == deterministic_answer(region, override).strip()

    crops = allowed_plants(region, override)
    bad_plants = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in FALSE_FRIENDS:
            continue
        for stem, canonical in PLANT_STEMS.items():
            if stem in word and canonical not in crops:
                bad_plants.add(f"{word} (implied {canonical}, allowed={sorted(crops)})")
    if is_fallback:
        bad_plants.add("FALLBACK: output is the deterministic template, "
                       "inference is NOT running")
    return bad_nums, bad_plants, text


def main():
    regions = sys.argv[1:] or sorted(FROST_DATES)
    passes = [("real", None), ("spring", SPRING_DAY)]
    failed = 0
    total = 0
    for label, pinned in passes:
        set_today(pinned)
        print(f"=== pass: {label}" + (f" ({pinned})" if pinned else " (real today)") + " ===")
        for region in regions:
            total += 1
            try:
                bad_n, bad_p, text = audit(region)
            except Exception as e:
                print(f"  ERROR {region}: {type(e).__name__}: {e}")
                failed += 1
                continue
            if bad_n or bad_p:
                failed += 1
                print(f"FAIL {region}")
                if bad_n:
                    print(f"     invented numbers: {sorted(bad_n)}")
                if bad_p:
                    for p in bad_p:
                        print(f"     invented plant:   {p}")
                print(f"     text: {text[:160]}")
            else:
                first = text.strip().split("\n")[0][:60]
                print(f"ok   {region:16} {first}")
        set_today(None)
    print()
    if failed:
        print(f"FAIL - {failed}/{total} generations produced unverifiable output")
        raise SystemExit(1)
    print(f"ok - {total} generations: every number and plant traceable to "
          f"the Python facts")


if __name__ == "__main__":
    main()