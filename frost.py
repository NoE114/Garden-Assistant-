"""Frost-date lookup and planting-window math.

Dates are approximate: they follow the commonly published 50%-probability
spring/fall frost maps (NOAA climate normals, as summarized in almanacs and
extension-service planting calendars). Treat them as a starting point, not a
forecast -- a single cold night in your own yard beats any table here.

Why not ask the model for this? Date arithmetic is exactly what small models
get wrong, and a wrong frost date silently corrupts every recommendation built
on top of it. So the dates and windows are computed here, deterministically,
and only the "what do I actually do with this" step goes to the LLM.
"""

import difflib
from datetime import date, timedelta

# A single override for "what day is it", used by the --today flag and by the
# tests to exercise spring and autumn in one run. Every date computation goes
# through current_date(); nothing else calls date.today() directly.
_TODAY_OVERRIDE: date | None = None


def set_today(value: date | str | None) -> None:
    """Pin the current date (tests / --today). None restores the real clock."""
    global _TODAY_OVERRIDE
    if value is None:
        _TODAY_OVERRIDE = None
    elif isinstance(value, str):
        _TODAY_OVERRIDE = date.fromisoformat(value)
    else:
        _TODAY_OVERRIDE = value


def current_date() -> date:
    return _TODAY_OVERRIDE or date.today()

# region -> (spring last-frost month/day, fall first-frost month/day)
# "Spring" = last expected frost of the year (safe transplant/sole-sow date).
# "Fall"   = first expected frost of the year (deadline for fall crops).
FROST_DATES = {
    "alaska":    ((5, 15), (9, 15)),
    "arizona":   ((3, 15), (11, 15)),
    "california": ((2, 15), (11, 30)),
    "colorado":  ((5, 10), (10, 5)),
    "connecticut": ((5, 5), (10, 10)),
    "florida":   ((2, 15), (11, 30)),
    "georgia":   ((4, 5), (11, 5)),
    "idaho":     ((5, 20), (9, 25)),
    "illinois":  ((4, 25), (10, 10)),
    "iowa":      ((5, 5), (10, 5)),
    "kansas":    ((4, 25), (10, 10)),
    "maine":     ((5, 20), (10, 5)),
    "maryland":  ((4, 10), (10, 20)),
    "massachusetts": ((5, 10), (10, 10)),
    "michigan":  ((5, 20), (10, 5)),
    "minnesota": ((5, 15), (10, 1)),
    "missouri":  ((4, 20), (10, 10)),
    "montana":   ((5, 25), (9, 25)),
    "nebraska":  ((5, 10), (10, 5)),
    "nevada":    ((4, 20), (10, 20)),
    "new hampshire": ((5, 20), (10, 5)),
    "new jersey": ((4, 10), (10, 20)),
    "new mexico": ((4, 25), (10, 10)),
    "new york":  ((5, 5), (10, 10)),
    "north carolina": ((4, 5), (11, 1)),
    "north dakota": ((5, 20), (9, 20)),
    "ohio":      ((5, 5), (10, 10)),
    "oklahoma":  ((4, 20), (10, 15)),
    "oregon":    ((4, 20), (10, 20)),
    "pennsylvania": ((5, 1), (10, 10)),
    "south carolina": ((4, 1), (11, 5)),
    "south dakota": ((5, 15), (9, 25)),
    "tennessee": ((4, 10), (10, 20)),
    "texas":     ((3, 15), (11, 20)),
    "utah":      ((5, 20), (10, 5)),
    "vermont":   ((5, 20), (10, 5)),
    "virginia":  ((4, 10), (10, 20)),
    "washington": ((4, 15), (10, 20)),
    "wisconsin": ((5, 15), (10, 5)),
    "wyoming":   ((5, 25), (9, 25)),
}

# Frost tolerance in whole days before the last spring frost / after the first
# fall frost. Young transplants are the least cold-tolerant thing in the garden,
# so these sit a few days inside the safe window rather than on the line.
COLD_TOLERANT = 28   # brassicas, peas, spinach, lettuce
SEMI_HARDY = 14      # beets, carrots, chard, cilantro
TENDER = 0            # tomatoes, peppers, cucumbers, squash, basil

# crop -> (days_to_maturity, frost_tolerant, lead_days_before_last_frost)
#
# frost_tolerant crops survive light frost, so they are allowed to finish
# FROST_GRACE_DAYS *past* first fall frost. A late-sown cabbage will frost-hit
# but still ripen; a late tomato will not. The flag is what lets the tool say
# "plant this now" in early autumn without lying about a tender crop.
#
# The third field is how early the crop may go in relative to last spring
# frost: brassicas shrug off -4C, but a beet seedling does not, so roots get a
# shorter lead time even though they are frost-tolerant later on.
CROPS = {
    "broccoli": (55, True,  COLD_TOLERANT),
    "kale":     (55, True,  COLD_TOLERANT),
    "peas":     (55, True,  COLD_TOLERANT),
    "spinach":  (55, True,  COLD_TOLERANT),
    "lettuce":  (55, True,  COLD_TOLERANT),
    "beets":    (55, True,  SEMI_HARDY),
    "carrots":  (55, True,  SEMI_HARDY),
    "chard":    (55, True,  SEMI_HARDY),
    "cilantro": (55, True,  SEMI_HARDY),
    "tomatoes": (65, False, TENDER),
    "peppers":  (65, False, TENDER),
    "cucumbers": (65, False, TENDER),
    "basil":    (65, False, TENDER),
    "squash":   (65, False, TENDER),
}

# Days past first fall frost that a frost-tolerant crop may still be growing.
FROST_GRACE_DAYS = 42

# Which group in `status()` each crop belongs to. This is presentation, not
# agronomy: it decides which prompt list the crop is shown under.
CROP_GROUPS = {
    "hardy_now":  ("broccoli", "kale", "peas", "spinach", "lettuce"),
    "tender_now": ("tomatoes", "peppers", "cucumbers", "basil", "squash"),
    "root_now":   ("beets", "carrots", "chard", "cilantro"),
}


def resolve(region: str) -> str:
    """Normalize free-text user input to a region key."""
    key = region.strip().lower()
    if not key:
        # Guard first: "alaska".startswith("") is True, so an empty argument
        # used to fall through the prefix loop and silently resolve to the
        # first key in the table.
        raise KeyError("no region given. Expected a state name, e.g. 'iowa'.")
    if key in FROST_DATES:
        return key
    # Accept common variants: "New York City" -> "new york", "N. Dakota" -> ...
    for known in FROST_DATES:
        if key.startswith(known) or known.startswith(key):
            return known
    close = difflib.get_close_matches(key, FROST_DATES, n=1, cutoff=0.7)
    hint = f" Did you mean {close[0]}?" if close else (
        f" Known regions: {', '.join(sorted(FROST_DATES))}"
    )
    raise KeyError(f"unknown region {region!r}.{hint}")


def frost_dates(region: str, override: tuple | None = None) -> tuple[date, date]:
    """(last spring frost, first fall frost) for the current growing season.

    `override` replaces the regional table entirely -- used by --last-frost /
    --first-frost for anyone whose actual yard disagrees with the table.

    Both dates belong to the same season, so they shift together. After fall
    frost has passed we report the *upcoming* season -- otherwise spring is in
    the past, fall is next year, and the gap between them is nonsense.
    """
    today = current_date()
    if override is not None:
        spring, fall = override
        # An override is used verbatim, but the same rollover rule applies: once
        # the season's first fall frost has passed, both dates belong to the
        # next season. Without this a stale pair (someone typing last year's
        # dates) reported days_until_last_frost of -158 AND claimed every crop
        # was sowable, because the closed season was never re-based.
        if today > fall:
            spring, fall = spring.replace(year=fall.year + 1), fall.replace(year=fall.year + 1)
        return spring, fall
    spring_md, fall_md = FROST_DATES[resolve(region)]
    year = today.year
    spring, fall = date(year, *spring_md), date(year, *fall_md)
    if today > fall:
        spring, fall = date(year + 1, *spring_md), date(year + 1, *fall_md)
    return spring, fall


def frost_free_days(region: str, override: tuple | None = None) -> int:
    """Days between last spring frost and first fall frost."""
    spring, fall = frost_dates(region, override)
    return (fall - spring).days


def safe_start(region: str, tolerance_days: int, override: tuple | None = None) -> date:
    """Earliest date to sow/transplant something with this cold tolerance.

    Tolerating `tolerance_days` of frost means the seed goes in that many days
    BEFORE the last spring frost -- not after. That is the whole trick behind
    planting by frost date rather than by calendar date.
    """
    spring, _ = frost_dates(region, override)
    return spring - timedelta(days=tolerance_days)


def grow_window(region: str, days_to_maturity: int, tolerance_days: int = TENDER,
                override: tuple | None = None) -> bool:
    """Can a crop with this maturity actually finish before first fall frost?"""
    _, fall = frost_dates(region, override)
    start = safe_start(region, tolerance_days, override)
    return start + timedelta(days=days_to_maturity) <= fall


def plantable_now(region: str, days_to_maturity: int, tolerance_days: int = TENDER,
                  override: tuple | None = None, frost_tolerant: bool = False) -> bool:
    """Should this crop go in the ground today?

    "Would it finish" is not the same question as "can I still plant it". By
    October a 65-day tender crop still has time to mature somewhere warm, but
    the window to sow it closed months ago. A crop is only worth recommending
    if its cold-tolerant start date has arrived AND it can still finish.

    Both conditions are checked from TODAY. The finish test used to count from
    safe_start, which for any open window is a date in the past -- so it
    answered "yes" for crops with no realistic chance of ripening. Texas in
    mid-October has ~41 days before first fall frost and was recommending
    65-day tomatoes: plant them, and the frost kills them unripe.

    `frost_tolerant` crops get FROST_GRACE_DAYS past first fall frost, because
    they survive light frost and can finish late. Tender crops get nothing.
    """
    start = safe_start(region, tolerance_days, override)
    _, fall = frost_dates(region, override)
    today = current_date()
    deadline = fall + (timedelta(days=FROST_GRACE_DAYS) if frost_tolerant
                       else timedelta(0))
    return start <= today and today + timedelta(days=days_to_maturity) <= deadline


def crop_plantable_now(region: str, crop: str, override: tuple | None = None) -> bool:
    """plantable_now() driven by the crop's entry in CROPS."""
    maturity, tolerant, lead = CROPS[crop]
    return plantable_now(region, maturity, lead, override, tolerant)


def validate_override(last_frost, first_frost) -> tuple[date, date]:
    """Validate a user-supplied (last_frost, first_frost) pair.

    Raises ValueError with a message meant to be shown to the user verbatim.
    These checks catch the mistakes that actually happen: swapped arguments,
    a typo'd year, or a pair that describes no real growing season.
    """
    last_frost = parse_iso(last_frost, "--last-frost")
    first_frost = parse_iso(first_frost, "--first-frost")

    # Northern-hemisphere assumption, consistent with the table above.
    if not 3 <= last_frost.month <= 6:
        raise ValueError(
            f"--last-frost {last_frost} is not a spring date. "
            f"Last spring frost should fall between March and June."
        )
    if not 8 <= first_frost.month <= 12:
        raise ValueError(
            f"--first-frost {first_frost} is not a fall date. "
            f"First fall frost should fall between August and December."
        )
    if first_frost <= last_frost:
        raise ValueError(
            f"--first-frost {first_frost} is not after --last-frost {last_frost}. "
            f"Check that the two dates are not swapped."
        )
    length = (first_frost - last_frost).days
    if not 30 <= length <= 365:
        raise ValueError(
            f"Season length of {length} days is out of range "
            f"(expected 30-365). Check the year on both dates."
        )
    return last_frost, first_frost


def parse_iso(value: str, flag: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(f"{flag} must be a date in YYYY-MM-DD form, got {value!r}")


def status(region: str, override: tuple | None = None) -> dict:
    """Everything the CLI and the prompt need, computed once."""
    spring, fall = frost_dates(region, override)
    today = current_date()
    groups = {
        name: [c for c in crops if crop_plantable_now(region, c, override)]
        for name, crops in CROP_GROUPS.items()
    }
    # Garlic is planted in FALL and harvested next summer. It ignores the
    # spring window entirely, so it stays available after the season closes.
    # Without this, an October query in a cold state answers "nothing", which
    # is technically true for spring crops and useless to an actual gardener.
    fall_planted = ["garlic"] if spring > today else []
    return {
        "region": resolve(region),
        "today": today.isoformat(),
        "last_spring_frost": spring.isoformat(),
        "first_fall_frost": fall.isoformat(),
        "frost_free_days": (fall - spring).days,
        "days_since_spring_frost": (today - spring).days,
        "days_until_fall_frost": (fall - today).days,
        "days_until_last_frost": (spring - today).days,
        "in_season": spring <= today <= fall,
        "hardy_now": groups["hardy_now"],
        "tender_now": groups["tender_now"],
        "root_now": groups["root_now"],
        "fall_planted": fall_planted,
    }


# Winter jobs. When nothing can go in the ground, "Nothing" is a useless
# answer -- a gardener in Iowa in October has real work to do. These are fixed
# chores, not derived from the model, so the list is always complete.
OFFSEASON_TASKS = [
    "plant garlic cloves 6 inches apart for next summer",
    "clear spent plants and roots out of the beds",
    "add a layer of compost or leaves over the empty beds",
    "order seeds for next spring while varieties are in stock",
    "check stored potatoes and onions for rot",
]


def offseason_plan(region: str, override: tuple | None = None) -> list[str]:
    """Concrete jobs for a region with nothing sowable right now.

    Always returns work, never an empty list, so the output can never be the
    single word "Nothing".

    Garlic is included only when `fall_planted` agrees. Hardcoding it here used
    to contradict status() for any region whose season was still open, and the
    model would faithfully repeat a crop the prompt's own facts had excluded.
    """
    jobs = []
    if status(region, override)["fall_planted"]:
        jobs.append("plant garlic cloves 6 inches apart for next summer")
    jobs += [t for t in OFFSEASON_TASKS if not t.startswith("plant garlic")]
    return jobs


def checklist(region: str, override: tuple | None = None) -> dict:
    """Facts for a one-page field checklist. Model never fills these in.

    Exactly 3 things to plant, 1 thing to check, 1 thing to look at -- sized
    for a phone screen in a garden, not a paragraph.
    """
    s = status(region, override)
    plant_now = s["hardy_now"] + s["root_now"] + s["tender_now"] + s["fall_planted"]

    if len(plant_now) >= 3:
        plants = plant_now[:3]
    else:
        # Out of season: list the crops the gardener is actually planning, so
        # the checklist still names three things rather than padding with junk.
        planned = ["garlic", "cover crop", "compost for beds"]
        plants = (plant_now + [p for p in planned if p not in plant_now])[:3]

    return {
        "region": s["region"],
        "date": s["today"],
        "plants": plants,
        "check": "soil: dig 5cm down and squeeze -- is it crumbly, not smeared?",
        "observe": "find one pollinator on a flower, or one pest on a leaf",
        "wait_for": s["last_spring_frost"],
        "in_season": s["in_season"],
        "fall_frost": s["first_fall_frost"],
    }