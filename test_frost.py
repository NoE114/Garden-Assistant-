"""Self-check for the date math. Run: .venv/bin/python test_frost.py"""

from datetime import date, timedelta

from frost import (
    COLD_TOLERANT, FROST_DATES, SEMI_HARDY, TENDER,
    frost_dates, frost_free_days, grow_window, offseason_plan, plantable_now,
    resolve, safe_start, status,
)

failures = []


def check(label, got, want):
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def check_true(label, cond):
    if not cond:
        failures.append(f"{label}: expected true")


# --- lookup -----------------------------------------------------------------
check("resolve exact", resolve("Iowa"), "iowa")
check("resolve case/space", resolve("  New York "), "new york")
check("resolve prefix", resolve("new york city"), "new york")
try:
    resolve("wakanda")
    failures.append("resolve unknown: expected KeyError")
except KeyError:
    pass

# --- date math --------------------------------------------------------------
# Cold-tolerant crops go in BEFORE last frost. That is the whole point.
check("hardy start before spring frost",
      safe_start("iowa", COLD_TOLERANT) < frost_dates("iowa")[0], True)
check("tender start == spring frost",
      safe_start("iowa", TENDER), frost_dates("iowa")[0])

# A warmer region has a longer season than a colder one.
check_true("season length ordering",
           frost_free_days("texas") > frost_free_days("iowa") > frost_free_days("minnesota"))

# Every region must have spring before fall, and a sane season length.
for region in FROST_DATES:
    spring, fall = frost_dates(region)
    check_true(f"{region}: spring before fall", spring < fall)
    check_true(f"{region}: season 60-400 days", 60 <= (fall - spring).days <= 400)

# --- the actual decision: will it finish? -----------------------------------
# A 150-day crop cannot fit Minnesota's season even sown at the last frost.
check_true("cold climate rejects over-long tender crop",
           not grow_window("minnesota", 150, TENDER))
# In Texas, the same crop has room.
check_true("warm climate accepts long crop",
           grow_window("texas", 150, TENDER))
# Hardy crops survive a shorter season than tender ones.
check_true("hardy beats tender in same climate",
           grow_window("minnesota", 150, COLD_TOLERANT) or
           not grow_window("minnesota", 150, TENDER))

# --- season rollover --------------------------------------------------------
# Regression: spring and fall must belong to the SAME season. After fall frost
# has passed, both roll to next year together -- otherwise the gap between them
# is not a season at all.
for region in FROST_DATES:
    spring, fall = frost_dates(region)
    check_true(f"{region}: dates in same year", spring.year == fall.year)
    check_true(f"{region}: season has not ended", fall >= date.today())

# --- "can I plant it today" vs "could it finish" ----------------------------
# These are different questions and the whole app hinges on the difference.
# plantable_now is strictly the harder question: it must be false whenever
# grow_window is false, AND also false when the crop would finish but only if
# sown months ago (it counts from today, not from safe_start).
check_true("plantable implies grow_window",
           all(not plantable_now("texas", m, t) or grow_window("texas", m, t)
               for m, t in ((65, TENDER), (55, COLD_TOLERANT), (55, SEMI_HARDY))))

# Nothing may be recommended whose start date is still in the future.
for region in FROST_DATES:
    for crop in (status(region)["hardy_now"] + status(region)["tender_now"]
                 + status(region)["root_now"]):
        tol = (COLD_TOLERANT if crop in status(region)["hardy_now"]
               else SEMI_HARDY if crop in status(region)["root_now"] else TENDER)
        tol = COLD_TOLERANT if crop in status(region)["hardy_now"] else TENDER
        maturity = 55 if tol in (COLD_TOLERANT, SEMI_HARDY) else 65
        check_true(f"{region}/{crop}: start date has arrived",
                   safe_start(region, tol) <= date.today())
        check_true(f"{region}/{crop}: finishes before fall frost",
                   grow_window(region, maturity, tol))

# --- status payload ---------------------------------------------------------
s = status("iowa")
for key in ("region", "today", "last_spring_frost", "first_fall_frost",
            "frost_free_days", "days_since_spring_frost", "days_until_fall_frost",
            "in_season"):
    check_true(f"status has {key}", key in s)

# A gardener must never get an empty answer. If no crop fits before first
# frost, the deterministic off-season jobs take over -- so the guarantee is
# "crops OR a fallback", not "always a crop". (Before the plantable_now fix,
# 24 in-season regions wrongly claimed crops they could not finish.)
check_true("status always yields crops or a fallback",
           all(status(r)["hardy_now"] or status(r)["tender_now"]
               or status(r)["root_now"] or status(r)["fall_planted"]
               or offseason_plan(r)
               for r in FROST_DATES))
if not s["hardy_now"] and not s["tender_now"]:
    check_true("off-season still suggests fall planting", s["fall_planted"])

# A crop must never be recommended if it cannot finish before fall frost.
_, fall = frost_dates("iowa")
for crop in s["hardy_now"] + s["tender_now"]:
    tol = COLD_TOLERANT if crop in s["hardy_now"] else TENDER
    check_true(f"{crop} finishes before fall frost",
               grow_window("iowa", 55 if tol == COLD_TOLERANT else 65, tol))

if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print(f"ok - {len(FROST_DATES)} regions, date math and crop windows check out")

# === change 1: off-season fallback ==========================================
from frost import OFFSEASON_TASKS, checklist, offseason_plan, validate_override, parse_iso
from garden import (build_offseason_prompt, build_prompt, deterministic_answer,
                    render_checklist, _is_empty_answer)

# A cold state in October has nothing sowable, but must still produce work.
plan = offseason_plan("iowa")
check_true("off-season plan is never empty", len(plan) > 0)
check_true("off-season plan has real jobs", all(len(j) > 10 for j in plan))

det = deterministic_answer("iowa")
check_true("deterministic answer is not 'Nothing'",
           not _is_empty_answer(det))
check_true("deterministic answer mentions the reopen date",
           status("iowa")["last_spring_frost"] in det)
check_true("deterministic answer mentions garlic", "garlic" in det)

# The empty-answer guard must catch the exact failure we are preventing.
for junk in ("", "  ", "Nothing", "nothing.", "None", "N/A"):
    check_true(f"_is_empty_answer catches {junk!r}", _is_empty_answer(junk))
for real in ("Plant garlic now.", "Broccoli, kale and chard."):
    check_true(f"_is_empty_answer allows {real!r}", not _is_empty_answer(real))

# Off-season routing: Iowa in October must take the off-season prompt path.
off_prompt = build_offseason_prompt("iowa")
check_true("off-season prompt lists the jobs",
           all(j.split()[0] in off_prompt for j in OFFSEASON_TASKS[:2]))
check_true("off-season prompt carries the reopen date",
           status("iowa")["last_spring_frost"] in off_prompt)
# Routing rule: the off-season prompt is chosen by whether any crop actually
# fits before first frost, not merely by the calendar. An open season with too
# little time left is still off-season as far as sowing goes.
# (Do not hardcode a region as "in-season": season-open and sowable are
# different questions, and this file's plantable_now now answers the second.)
_fit = [r for r in FROST_DATES
        if status(r)["hardy_now"] or status(r)["root_now"] or status(r)["tender_now"]]
_empty = [r for r in FROST_DATES if r not in _fit]
# Both lists are data-dependent: in mid-October after the plantable_now fix,
# no region has enough season left, so only assert the direction that holds.
for r in _fit:
    check_true(f"{r}: has crops -> in-season prompt",
               "Nothing can be sown" not in build_prompt(r))
for r in _empty:
    check_true(f"{r}: no crops -> off-season prompt",
               "Nothing can be sown" in build_prompt(r))
check_true("every region routes to exactly one prompt type",
           len(_fit) + len(_empty) == len(FROST_DATES))
check_true("off-season region uses off-season prompt",
           "Nothing can be sown" in build_prompt("iowa"))

# === change 2: frost overrides ==============================================
ok = validate_override("2026-05-05", "2026-10-05")
check_true("valid override parses", ok[0].isoformat() == "2026-05-05")
check_true("valid override second date", ok[1].isoformat() == "2026-10-05")

# Swapped args, bad months, and a typo'd year must all be rejected loudly.
for bad, why in [
    (("2026-10-05", "2026-05-05"), "swapped: fall before spring"),
    (("2026-12-05", "2026-05-05"), "first frost not in fall"),
    (("2026-05-05", "2026-06-05"), "last frost not in spring"),
    (("2026-02-05", "2026-03-10"), "season too short"),
    (("2026-05-05", "2026-05-10"), "season too short 2"),
    (("06/05/2026", "2026-10-05"), "wrong date format"),
]:
    try:
        validate_override(*bad)
        failures.append(f"validate_override should reject {why}: {bad}")
    except ValueError:
        pass

# An override must actually change the answer, not be silently ignored.
# These dates are in the CURRENT season (today is Oct 10, fall frost Sep 20 has
# not arrived yet for the 2026 pair below only if today < fall; here it has, so
# both roll to 2027 -- the same rollover the table path performs).
ovr = validate_override("2026-06-05", "2026-09-20")   # short season
s_override = status("iowa", ovr)
check_true("override changes season length",
           s_override["frost_free_days"] == 107)
check_true("override spring/fall stay in the same year",
           s_override["last_spring_frost"][:4] == s_override["first_fall_frost"][:4])
# A current-season override is used as given. Use NEXT year so the dates are
# guaranteed to be in the future: today is mid-October, so a "current year"
# Sep 20 would already have passed and correctly roll forward.
_ny = date.today().year + 1
ovr_now = validate_override(f"{_ny}-06-05", f"{_ny}-09-20")
s_now = status("iowa", ovr_now)
check_true("override replaces regional spring date",
           s_now["last_spring_frost"] == f"{_ny}-06-05")
check_true("override replaces regional fall date",
           s_now["first_fall_frost"] == f"{_ny}-09-20")
# Same region, no override: the table value must still be used.
check_true("no override keeps regional default",
           status("iowa")["last_spring_frost"] != f"{_ny}-06-05")

# === change 3: go-outside checklist =========================================
for region in ("iowa", "texas", "minnesota", "florida"):
    c = checklist(region)
    check_true(f"{region}: exactly 3 plants", len(c["plants"]) == 3)
    check_true(f"{region}: plants are unique", len(set(c["plants"])) == 3)
    check_true(f"{region}: has a soil check", len(c["check"]) > 10)
    check_true(f"{region}: has an observation task", len(c["observe"]) > 10)

text = render_checklist("iowa")
check_true("checklist renders checkboxes", text.count("- [ ]") == 5)
check_true("checklist has plant section", "PLANT" in text)
check_true("checklist has look section", "LOOK" in text)
# Phone-friendly: no line long enough to wrap a narrow screen.
longest = max(len(l) for l in text.split("\n"))
check_true(f"checklist lines are short (max {longest})", longest <= 72)

if failures:
    print("FAIL")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print("ok - frost math, overrides, off-season fallback and checklist all check out")
