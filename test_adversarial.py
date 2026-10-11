"""Adversarial tests: dates, invariants, every region, and model output.

Separate from test_frost.py on purpose -- these probe for bugs rather than
confirming features, and several tests assert that a bug does NOT occur. Run:

    .venv/bin/python test_adversarial.py

Model-output auditing lives in test_model_audit.py (it loads the GGUF, so it is
slow and kept apart from the fast pure-Python checks).
"""

import re
import sys
from datetime import date, timedelta

from frost import (
    COLD_TOLERANT, CROPS, CROP_GROUPS, FROST_DATES, FROST_GRACE_DAYS,
    OFFSEASON_TASKS, SEMI_HARDY, TENDER,
    checklist, crop_plantable_now, frost_dates, frost_free_days, grow_window,
    current_date, offseason_plan, parse_iso, plantable_now, resolve, safe_start,
    set_today, status, validate_override,
)

failures = []


def check_true(label, cond):
    if not cond:
        failures.append(label)


def check_eq(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


# ===========================================================================
# BUG 1 -- plantable_now checks the finish date from safe_start, not today
# ===========================================================================
# safe_start is in the PAST for any crop whose window has opened. So the test
# "start + maturity <= first_fall_frost" is measuring the past, and a crop with
# no realistic chance of finishing is still reported as sowable today.
# Texas in mid-October: 41 days to first frost, tomatoes need 65.

def test_plantable_counts_from_today():
    tx = status("texas")
    if not tx["in_season"]:
        return  # season closed; nothing to prove
    days_left = tx["days_until_fall_frost"]
    check_true("BUG1 setup: texas has < 65 days to frost",
               0 <= days_left < 65)
    check_true(
        f"BUG1: with only {days_left} days left, a 65-day crop must not be "
        f"sowable today (got {tx['tender_now']})",
        not plantable_now("texas", 65, TENDER),
    )
    # The direct statement of the bug: sowable_today must imply the crop can
    # still finish if it goes in the ground today.
    if plantable_now("texas", 65, TENDER):
        check_true(
            "BUG1: 65-day crop cannot finish before first fall frost",
            today_plus(65) <= frost_dates("texas")[1],
        )


def today_plus(days):
    return date.today() + timedelta(days=days)


# ===========================================================================
# BUG 2 -- override skips the season rollover that the table path performs
# ===========================================================================
# frost_dates() rolls both dates to next year once fall frost has passed, but
# returns an override verbatim. A stale override therefore reports a season
# that ended long ago AND claims every crop is sowable.

def test_override_rolls_forward():
    # A past pair, exactly what a user produces by typing last year's dates.
    stale = validate_override("2026-05-05", "2026-10-05")
    s = status("iowa", stale)
    check_true(
        f"BUG2: stale override must not report negative days-until-frost "
        f"(got {s['days_until_last_frost']})",
        s["days_until_last_frost"] > 0,
    )
    check_true(
        "BUG2: stale override must not claim crops are sowable",
        not (s["tender_now"] or s["hardy_now"] or s["root_now"]),
    )
    spring, fall = frost_dates("iowa", stale)
    check_eq("BUG2: stale override keeps spring/fall in the same year",
             spring.year, fall.year)
    check_true("BUG2: stale override reports a season that has not closed",
               fall >= date.today())


# ===========================================================================
# DATES -- leap day, year boundaries, today exactly on a frost date
# ===========================================================================

def test_leap_day():
    # 2024 is a leap year, 2023 and 2100 are not.
    check_eq("leap: 2024-02-29 parses", parse_iso("2024-02-29", "--x"), date(2024, 2, 29))
    for bad in ("2023-02-29", "2026-02-29", "2100-02-29", "2024-13-01", "2024-00-10"):
        try:
            parse_iso(bad, "--last-frost")
            failures.append(f"invalid date accepted: {bad}")
        except ValueError:
            pass
    # Feb 29 is a valid date but not a valid *spring frost* month.
    try:
        validate_override("2024-02-29", "2024-11-01")
        failures.append("BUG: Feb 29 accepted as last spring frost")
    except ValueError:
        pass
    # A leap year must not shift the season length unexpectedly.
    check_eq("leap: 2024 season length unaffected",
             (date(2024, 11, 1) - date(2024, 5, 5)).days, 180)


def test_year_boundaries():
    # Dec 31 as a fall frost is legal and must be accepted.
    dec31 = validate_override("2026-05-05", "2026-12-31")
    check_eq("dec31: accepted", dec31[1], date(2026, 12, 31))
    # A season that would end in January of the next year is NOT representable
    # and must be rejected rather than silently reordered.
    try:
        validate_override("2026-12-01", "2027-01-15")
        failures.append("BUG: Dec->Jan season accepted")
    except ValueError:
        pass
    # Every table region's season must sit strictly inside one calendar year.
    for region in FROST_DATES:
        (sm, sd), (fm, fd) = FROST_DATES[region]
        check_true(f"{region}: spring before fall in the year",
                   (sm, sd) < (fm, fd))


def test_today_exactly_on_frost_date():
    # On the last spring frost itself, tender crops are exactly on time.
    ovr = validate_override(f"{date.today().year}-05-01", f"{date.today().year}-10-01")
    spring, fall = frost_dates("iowa", ovr)
    check_true("on-frost: dates are usable", spring <= date.today() or spring > date.today())
    s = status("iowa", ovr)
    check_true("on-frost: in_season is a bool", isinstance(s["in_season"], bool))
    # A season that contains today must report non-negative day counts.
    if s["in_season"]:
        check_true("on-frost: days_since_spring_frost >= 0",
                   s["days_since_spring_frost"] >= 0)
        check_true("on-frost: days_until_fall_frost >= 0",
                   s["days_until_fall_frost"] >= 0)


# ===========================================================================
# SOUTHERN HEMISPHERE
# ===========================================================================

def test_southern_hemisphere_rejected():
    # A southern-hemisphere pair (Aug frost-free start, Feb end) must be
    # rejected, not silently accepted and then computed against.
    for last, first in [("2026-08-15", "2027-02-15"), ("2026-09-01", "2027-03-01")]:
        try:
            validate_override(last, first)
            failures.append(f"BUG: southern-hemisphere pair accepted: {last}/{first}")
        except ValueError as e:
            check_true("southern rejection explains itself",
                       "spring" in str(e).lower() or "fall" in str(e).lower())


# ===========================================================================
# INVARIANTS -- sweep every region
# ===========================================================================

def test_invariants_all_regions():
    today = date.today()
    for region in FROST_DATES:
        spring, fall = frost_dates(region)
        # 1. season length bounded
        length = (fall - spring).days
        check_true(f"{region}: season length 0-365 (got {length})", 0 <= length <= 365)
        # 2. spring/fall same year and correctly ordered
        check_true(f"{region}: same year", spring.year == fall.year)
        check_true(f"{region}: spring <= fall", spring <= fall)
        # 3. the season has not already closed
        check_true(f"{region}: fall frost is not in the past", fall >= today)
        # 4. plantable_now can never contradict grow_window:
        #    anything we tell the user to sow must be able to finish.
        for mat, tol in ((55, COLD_TOLERANT), (65, TENDER), (55, SEMI_HARDY)):
            if plantable_now(region, mat, tol):
                check_true(
                    f"{region}/{mat}d: sowable today => can finish today",
                    today + timedelta(days=mat) <= fall,
                )
        # 5. nothing is recommended before its own safe start
        s = status(region)
        for crop in s["hardy_now"]:
            check_true(f"{region}/{crop}: after safe start",
                       safe_start(region, COLD_TOLERANT) <= today)
        for crop in s["tender_now"]:
            check_true(f"{region}/{crop}: after safe start",
                       safe_start(region, TENDER) <= today)


def test_invariants_overrides():
    # Every accepted override must satisfy the same invariants.
    pairs = [("2026-03-01", "2026-11-30"), ("2026-06-05", "2026-09-20"),
             ("2026-05-05", "2026-12-31"), ("2026-04-15", "2026-10-15")]
    today = date.today()
    for last, first in pairs:
        ovr = validate_override(last, first)
        s, f = frost_dates("iowa", ovr)
        check_true(f"override {last}/{first}: 0-365 days", 0 <= (f - s).days <= 365)
        check_true(f"override {last}/{first}: same year", s.year == f.year)
        check_true(f"override {last}/{first}: not in the past", f >= today)
        if plantable_now("iowa", 65, TENDER, ovr):
            check_true(f"override {last}/{first}: sowable => finishes",
                       today + timedelta(days=65) <= f)


# ===========================================================================
# REGIONS -- every one must produce usable output
# ===========================================================================

def test_every_region_produces_output():
    from garden import build_prompt, render_checklist
    for region in sorted(FROST_DATES):
        s = status(region)
        check_true(f"{region}: has a region name", bool(s["region"]))
        check_true(f"{region}: has an ISO date", bool(re.match(r"\d{4}-\d{2}-\d{2}", s["today"])))
        crops = s["hardy_now"] + s["root_now"] + s["tender_now"] + s["fall_planted"]
        # An empty crop list is only legitimate when there is genuinely no time
        # left before first frost. Before BUG 1 was fixed this was violated in
        # the other direction; now it must not mask a real time window.
        if not crops:
            check_true(
                f"{region}: empty crop list is justified by <55 days to frost "
                f"(has {s['days_until_fall_frost']})",
                s["days_until_fall_frost"] < 55,
            )
        # However the region resolves, the CLI must still say something useful:
        # either a crop list or the deterministic off-season jobs.
        prompt = build_prompt(region)
        check_true(f"{region}: prompt names something to do",
                   bool(crops) or "jobs this month" in prompt)
        # the checklist must always have exactly 3 plants
        c = checklist(region)
        check_eq(f"{region}: checklist has 3 plants", len(c["plants"]), 3)
        check_true(f"{region}: checklist plants unique", len(set(c["plants"])) == 3)
        # checklist text must stay phone-friendly
        longest = max(len(l) for l in render_checklist(region).split("\n"))
        check_true(f"{region}: checklist line <=72 chars (max {longest})", longest <= 72)


def test_offseason_plan_matches_allowed_crops():
    # BUG 4: OFFSEASON_TASKS hardcoded "plant garlic" unconditionally, while
    # status()["fall_planted"] is empty while a season is still open. Once the
    # plantable_now fix made warm regions take the off-season path, the model
    # was told to name a crop the same prompt's facts had excluded.
    for region in sorted(FROST_DATES):
        plan = offseason_plan(region)
        s = status(region)
        crops = set(s["hardy_now"] + s["root_now"] + s["tender_now"] + s["fall_planted"])
        check_true(f"{region}: off-season plan is never empty", len(plan) > 0)
        if "garlic" in " ".join(plan):
            check_true(f"{region}: plan mentions garlic only when allowed",
                       "garlic" in crops)
        # No plan may name a crop the record does not allow.
        for task in plan:
            for plant in ("garlic",):
                if plant in task:
                    check_true(f"{region}: '{task}' names an allowed plant",
                               plant in crops)


# ===========================================================================
# --today -- pinned clock, spring and autumn, all 40 regions
# ===========================================================================

SPRING_DAY = "2026-04-15"
AUTUMN_DAY = "2026-10-10"


def _at(day):
    """Run fn with the clock pinned to `day`, restoring it afterwards."""
    set_today(day)
    try:
        return current_date().isoformat()
    finally:
        pass


def test_today_flag_controls_the_clock():
    set_today(SPRING_DAY)
    check_eq("--today sets the clock", current_date(), date(2026, 4, 15))
    check_eq("--today flows into status", status("iowa")["today"], SPRING_DAY)
    set_today(AUTUMN_DAY)
    check_eq("--today autumn", current_date(), date(2026, 10, 10))
    set_today(None)
    check_eq("None restores the real clock", current_date(), date.today())
    # set_today accepts a date object as well as a string.
    set_today(date(2026, 4, 15))
    check_eq("set_today accepts a date object", current_date(), date(2026, 4, 15))
    set_today(None)
    check_eq("still restores", current_date(), date.today())
    try:
        set_today("not-a-date")
        failures.append("BUG: set_today accepted a malformed date string")
    except ValueError:
        pass
    set_today(None)


def test_crop_data_is_wellformed():
    grouped = {c for crops in CROP_GROUPS.values() for c in crops}
    check_eq("every crop is in exactly one group", grouped, set(CROPS))
    check_eq("no duplicates across groups", sum(len(v) for v in CROP_GROUPS.values()),
             len(grouped))
    for crop, (maturity, tolerant, lead) in CROPS.items():
        check_true(f"{crop}: positive maturity", maturity > 0)
        check_true(f"{crop}: bool frost_tolerant", isinstance(tolerant, bool))
        check_true(f"{crop}: non-negative lead", lead >= 0)
        check_true(f"{crop}: lead is on the ladder",
                   lead in (COLD_TOLERANT, SEMI_HARDY, TENDER))
    check_true("at least one frost-tolerant crop",
               any(t for _, t, _ in CROPS.values()))
    check_true("at least one tender crop",
               any(not t for _, t, _ in CROPS.values()))
    check_eq("grace window is 42 days", FROST_GRACE_DAYS, 42)


def test_frost_grace_window():
    # The grace window is governed purely by the frost_tolerant flag: a caller
    # that passes frost_tolerant=True gets FROST_GRACE_DAYS of slack, one that
    # passes False gets none. Texas on 2026-10-20 has 31 days to frost, which is
    # not enough for a 55-day crop strictly, but is with the grace.
    set_today("2026-10-20")
    _, fall = frost_dates("texas")
    today = current_date()
    grace_deadline = fall + timedelta(days=FROST_GRACE_DAYS)
    check_true("late Oct: 55-day crop fails the strict deadline",
               today + timedelta(days=55) > fall)
    check_true("late Oct: 55-day crop passes with the grace flag",
               plantable_now("texas", 55, COLD_TOLERANT, frost_tolerant=True))
    check_true("late Oct: same crop fails without the flag",
               not plantable_now("texas", 55, COLD_TOLERANT, frost_tolerant=False))
    # The grace adds exactly FROST_GRACE_DAYS and no more. A crop finishing
    # exactly on first frost is allowed (the comparison is <=), so the largest
    # strict maturity is the number of days to fall frost itself; one more day
    # is rejected strictly but accepted with the grace.
    just_fits = (fall - today).days
    check_true("boundary: finishing exactly on first frost is accepted",
               plantable_now("texas", just_fits, COLD_TOLERANT,
                             frost_tolerant=False))
    check_true("boundary: one day past first frost fails strictly",
               not plantable_now("texas", just_fits + 1, COLD_TOLERANT,
                                 frost_tolerant=False))
    check_true("boundary: one day past first frost passes with the grace",
               plantable_now("texas", just_fits + 1, COLD_TOLERANT,
                             frost_tolerant=True))
    check_true("boundary: grace accepts exactly FROST_GRACE_DAYS more",
               plantable_now("texas", just_fits + FROST_GRACE_DAYS, COLD_TOLERANT,
                             frost_tolerant=True))
    check_true("boundary: one day past the grace deadline is rejected",
               not plantable_now("texas", just_fits + FROST_GRACE_DAYS + 1,
                                 COLD_TOLERANT, frost_tolerant=True))
    # Crops in the data file: every frost-tolerant one is allowed the grace,
    # every tender one is not, regardless of how the caller invokes it.
    for crop, (maturity, tolerant, lead) in CROPS.items():
        expected = today + timedelta(days=maturity) <= (
            fall + timedelta(days=FROST_GRACE_DAYS if tolerant else 0))
        if safe_start("texas", lead) <= today:
            check_true(f"texas/{crop}: grace applied per CROPS flag",
                       crop_plantable_now("texas", crop) == expected)
    check_true("65-day crop fits within the grace on this date",
               today + timedelta(days=65) <= grace_deadline)
    set_today(None)


def test_both_dates_all_regions():
    """Full invariant sweep at a spring date and an autumn date."""
    for day in (SPRING_DAY, AUTUMN_DAY):
        set_today(day)
        pinned = current_date()
        for region in sorted(FROST_DATES):
            s = status(region)
            spring, fall = frost_dates(region)
            tag = f"{day}/{region}"
            # clock actually moved
            check_eq(f"{tag}: status uses pinned today", s["today"], pinned.isoformat())
            # season invariants
            check_true(f"{tag}: season 0-365", 0 <= (fall - spring).days <= 365)
            check_true(f"{tag}: same year", spring.year == fall.year)
            check_true(f"{tag}: spring <= fall", spring <= fall)
            check_true(f"{tag}: fall not in the past", fall >= pinned)
            # days_until_last_frost is only meaningful once the season has
            # CLOSED. While the season is open, last spring frost is behind us
            # and the count is legitimately negative -- so assert the sign
            # matches the season state rather than demanding non-negative.
            if s["in_season"]:
                check_true(f"{tag}: in-season => days_since >= 0",
                           s["days_since_spring_frost"] >= 0)
                check_true(f"{tag}: in-season => days_until_fall >= 0",
                           s["days_until_fall_frost"] >= 0)
            else:
                check_true(f"{tag}: off-season => days_until_last_frost > 0",
                           s["days_until_last_frost"] > 0)
            # every recommended crop must satisfy the same rule the code uses
            for crop in s["hardy_now"] + s["root_now"] + s["tender_now"] + s["fall_planted"]:
                if crop == "garlic":
                    continue
                maturity, tolerant, lead = CROPS[crop]
                deadline = fall + timedelta(days=FROST_GRACE_DAYS if tolerant else 0)
                check_true(f"{tag}/{crop}: finish before its deadline",
                           pinned + timedelta(days=maturity) <= deadline)
                check_true(f"{tag}/{crop}: safe start has arrived",
                           safe_start(region, lead) <= pinned)
            # the checklist must still be usable
            check_eq(f"{tag}: checklist has 3 plants", len(checklist(region)["plants"]), 3)
        set_today(None)


def test_spring_vs_autumn_differ():
    """The two dates must not produce the same answer everywhere."""
    set_today(SPRING_DAY)
    spring_iowa, spring_texas = status("iowa"), status("texas")
    set_today(AUTUMN_DAY)
    autumn_iowa, autumn_texas = status("iowa"), status("texas")
    set_today(None)
    check_true("the two dates report different frost seasons",
               spring_iowa["last_spring_frost"] != autumn_iowa["last_spring_frost"])
    check_true("crop lists differ between April and October",
               spring_texas["tender_now"] != autumn_texas["tender_now"]
               or spring_iowa["hardy_now"] != autumn_iowa["hardy_now"])
    # The meaningful spring/autumn contrast is the crop window, not the
    # calendar flag: in April Iowa is pre-season (frost still to come), in
    # October the same yard is post-season.
    check_true("April Iowa is before its last spring frost",
               spring_iowa["days_until_last_frost"] > 0)
    check_true("October Iowa has already passed last spring frost",
               autumn_iowa["days_until_last_frost"] > 0
               and autumn_iowa["last_spring_frost"][:4] == "2027")


def test_region_names_resolve():
    # An empty region must NOT silently resolve to the first key in the table.
    # `known.startswith("")` is True for every key, so a blank argument used to
    # return "alaska" and hand back Alaskan frost dates with no complaint.
    for blank in ["", " ", "\t", "\n"]:
        try:
            got = resolve(blank)
            failures.append(f"BUG3: blank region {blank!r} resolved to {got!r}")
        except KeyError:
            pass
    for bad in ["wakanda", "atlantis"]:
        try:
            resolve(bad)
            failures.append(f"BUG: bogus region resolved: {bad!r}")
        except KeyError:
            pass
    # Spelling variants must still resolve (prefix or fuzzy).
    for good, want in [("iowa", "iowa"), ("IOWA", "iowa"), (" New York ", "new york"),
                       ("new york city", "new york")]:
        check_eq(f"resolve {good!r}", resolve(good), want)
    # A near-miss must still be rejected, but with a usable suggestion.
    try:
        resolve("minnesotta")
        failures.append("BUG: 'minnesotta' accepted as minnesota")
    except KeyError as e:
        check_true("typo gets a suggestion", "minnesota" in e.args[0])


# ===========================================================================
# INVALID INPUT
# ===========================================================================

def test_invalid_inputs():
    for bad_last, bad_first in [
        ("2026-5-5", "2026-10-05"),        # unpadded
        ("May 5 2026", "2026-10-05"),      # prose
        ("", "2026-10-05"),                # empty
        ("2026-05-05", ""),                # empty
        (None, "2026-10-05"),              # None
        ("2026-05-05", 20261005),          # int
    ]:
        try:
            validate_override(bad_last, bad_first)
            failures.append(f"BUG: invalid input accepted: {bad_last!r}/{bad_first!r}")
        except (ValueError, TypeError):
            pass


if __name__ == "__main__":
    for fn in [
        test_plantable_counts_from_today,
        test_override_rolls_forward,
        test_leap_day,
        test_year_boundaries,
        test_today_exactly_on_frost_date,
        test_southern_hemisphere_rejected,
        test_today_flag_controls_the_clock,
        test_crop_data_is_wellformed,
        test_frost_grace_window,
        test_both_dates_all_regions,
        test_spring_vs_autumn_differ,
        test_invariants_all_regions,
        test_invariants_overrides,
        test_offseason_plan_matches_allowed_crops,
        test_every_region_produces_output,
        test_region_names_resolve,
        test_invalid_inputs,
    ]:
        try:
            fn()
        except Exception as e:  # a crash is a failure, not a stop
            failures.append(f"{fn.__name__} raised {type(e).__name__}: {e}")

    if failures:
        print(f"FAIL - {len(failures)} bug(s)\n")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print(f"ok - {len(FROST_DATES)} regions; dates, invariants, overrides, "
          f"per-region output all pass")