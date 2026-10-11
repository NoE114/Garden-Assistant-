"""Turn frost-date math into a planting plan, using a local open-weight model.

Everything factual -- the dates, the crop windows, what can physically mature
before first frost -- is computed in frost.py. This module only asks the model
to do the part that is actually a language task: explaining that to a person,
and adapting it to their situation.

That split is deliberate. A 1.7B model asked to do the date arithmetic itself
will confidently produce dates that contradict the tables above it. So we do not
ask it to. We hand it finished facts and ask for prose.

No API key, no network, no server. The GGUF runs on your machine.
"""

import argparse
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from frost import (checklist, offseason_plan, parse_iso, resolve, set_today,
                   status, validate_override)

MODEL_PATH = Path(__file__).parent / "models" / "Qwen3-1.7B-Q8_0.gguf"
MODEL_URL = ("https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/"
             "Qwen3-1.7B-Q8_0.gguf")
# Exact size of the file MODEL_URL serves. An interrupted download leaves a
# truncated file that still "exists", and llama.cpp then dies with a raw
# tensor-out-of-bounds traceback -- so we check the size before loading.
MODEL_BYTES = 1_834_426_016

SYSTEM = """You are a practical backyard gardener giving a short, practical answer.

Rules:
- The dates and crop list you are given are already calculated correctly. Never
  recalculate or contradict them. Treat them as ground truth.
- Never recommend a crop that is not in the provided lists.
- Be concrete and brief. No disclaimers about consulting a professional.
- If the lists are empty, say what to do instead this time of year.
- Plain text. No markdown headers, no bullet symbols."""

PROMPT = """Today is {today}. The gardener is in {region}.

{facts}

{season_note}

Write 3 short sentences:
1. What to plant right now.
2. What to wait for.
3. One concrete job to do this week.

Rules: name ONLY crops from the lists above. Do not invent other plants.
Plain text, no headings, no bullet points."""


OFFSEASON_PROMPT = """Today is {today}. The gardener is in {region}.

Nothing can be sown outdoors in {region} right now. The season reopens on
{next_start}, which is {days_until} days away.

Their jobs this month are already decided:
{jobs}

Rewrite that list as 3 short, friendly sentences for a gardener standing in
their yard. Keep every job. Mention {next_start} as the date things start
again.

Rules: do not name any crop that is not in the list above. Do not add new
tasks. Plain text, no headings, no bullet points."""


def build_offseason_prompt(region: str, override=None) -> str:
    """The deterministic fallback: the jobs and dates are fixed in Python, the
    model only turns them into sentences."""
    s = status(region, override)
    jobs = "\n".join(f"- {job}" for job in offseason_plan(region, override))
    return OFFSEASON_PROMPT.format(
        today=s["today"], region=s["region"], jobs=jobs,
        next_start=s["last_spring_frost"],
        days_until=s["days_until_last_frost"],
    )


def build_prompt(region: str, override=None) -> str:
    s = status(region, override)
    # Garlic is in fall_planted but is NOT sown outdoors -- it goes in as a
    # clove in autumn and sits under the soil all winter. So the off-season
    # path is chosen by the spring/summer crop lists being empty, not by the
    # whole record looking empty. Otherwise a cold October in Iowa, where the
    # only entry is garlic, would skip the fallback it exists to provide.
    if not s["hardy_now"] and not s["root_now"] and not s["tender_now"]:
        return build_offseason_prompt(region, override)
    # After the season closes every spring-crop list is empty. Garlic is still
    # plantable, so it is folded into the "plant now" line instead of a
    # separate "for next year" line -- keeping the two apart invited the model
    # to answer "nothing to plant, but plant garlic" in one breath.
    hardy_now = s["hardy_now"] + s["root_now"] + s["fall_planted"]
    lists = [
        f"- Tolerates frost, plant now: {', '.join(hardy_now) or 'nothing'}",
        f"- Tender, plant now: {', '.join(s['tender_now']) or 'nothing'}",
    ]
    if s["in_season"]:
        season_note = "The season is open."
        facts = "\n".join(lists + [
            f"- First fall frost: {s['first_fall_frost']}"
            f" ({s['days_until_fall_frost']} days from now)",
        ])
    else:
        # After the season closes, every crop list is empty and "days until
        # fall frost" counts into next year -- a number like 360 that reads as
        # nonsense and invited the model to say "wait 360 days". The useful
        # milestone is when the season reopens, so only that is shown.
        season_note = ("The season closed for this year, and they are planning for "
                       "the next one, which opens on:")
        facts = "\n".join(lists + [f"- Next season starts: {s['last_spring_frost']}"])
    return PROMPT.format(season_note=season_note, facts=facts,
                         region=s["region"], today=s["today"])


EMPTY_ANSWERS = {"nothing", "none", "n/a", "nothing to plant", "no"}


def _is_empty_answer(text: str) -> bool:
    """Did the model return nothing useful? Guards the off-season case, where
    'Nothing' is the one answer that would make the tool useless."""
    return (not text.strip()
            or text.strip().rstrip(".!?").strip().lower() in EMPTY_ANSWERS)


def deterministic_answer(region, override=None) -> str:
    """The answer with no model in the loop. Always says something real."""
    s = status(region, override)
    jobs = "; ".join(offseason_plan(region, override))
    return (f"Nothing can be sown outdoors in {s['region']} right now. "
            f"The season reopens on {s['last_spring_frost']}, "
            f"{s['days_until_last_frost']} days away. "
            f"This month: {jobs}.")


def advise(region: str, override=None, *, n_predict: int = 220, verbose: bool = False) -> str:
    # -C - resumes rather than restarting, so a partial file is never discarded
    # and a complete one is never silently overwritten with a shorter download.
    download_hint = f"download it with:  curl -L -C - -o {MODEL_PATH} {MODEL_URL}"
    if not MODEL_PATH.exists():
        raise SystemExit(f"model missing: {MODEL_PATH}\n{download_hint}")
    size = MODEL_PATH.stat().st_size
    if size != MODEL_BYTES:
        raise SystemExit(
            f"model file is incomplete or corrupt: {MODEL_PATH}\n"
            f"  on disk: {size:,} bytes\n  expected: {MODEL_BYTES:,} bytes\n"
            f"{download_hint}"
        )
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        # -1 = offload every layer this build can offload. The CPU-only build
        # documented in the README reports no GPU support, so this is a no-op
        # there and the run takes ~10s on a 12-core CPU; a CUDA build uses the
        # GPU automatically and needs no code change.
        n_gpu_layers=-1,
        verbose=verbose,
    )
    # Qwen3 "thinks" before answering by default and dumps the scratchpad into
    # the output. We want the answer, not the reasoning. The handler renders the
    # chat template; defining enable_thinking up front flips that branch off.
    formatter = Jinja2ChatFormatter(
        template="{% set enable_thinking = false %}"
                 + llm.metadata["tokenizer.chat_template"],
        eos_token="<|im_end|>",
        bos_token="",
    )
    rendered = formatter(llm=llm, messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(region, override)},
    ])
    result = llm.create_completion(
        rendered.prompt,
        max_tokens=n_predict,
        temperature=0.3,   # low: we want it to follow the given facts, not riff
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    text = result["choices"][0]["text"].strip()
    if _is_empty_answer(text):
        return deterministic_answer(region, override)
    return text


def render_checklist(region, override=None) -> str:
    """One page, short lines, checkboxes. No model -- this is printed straight
    from the computed facts so it stays readable on a phone in a garden."""
    c = checklist(region, override)
    lines = [f"Garden checklist - {c['region']} - {c['date']}", ""]
    lines.append("PLANT (3)")
    lines += [f"- [ ] {p}" for p in c["plants"]]
    lines.append("")
    lines.append("CHECK (1)")
    lines.append(f"- [ ] {c['check']}")
    lines.append("")
    lines.append("LOOK (1)")
    lines.append(f"- [ ] {c['observe']}")
    lines.append("")
    # This label is only true when the season is closed; in-season it would
    # print a past date and read as nonsense.
    if c["in_season"]:
        lines.append(f"Season open. Frost free until {c['fall_frost']}.")
    else:
        lines.append(f"Nothing outdoors until {c['wait_for']}.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        prog="garden.py", description="What to plant now, from local frost dates.")
    ap.add_argument("region", help="state or region, e.g. iowa")
    ap.add_argument("--last-frost", help="override last spring frost, YYYY-MM-DD")
    ap.add_argument("--first-frost", help="override first fall frost, YYYY-MM-DD")
    ap.add_argument("--checklist", action="store_true",
                    help="print a one-page field checklist instead of prose")
    ap.add_argument("--today", help="pretend today is this date, YYYY-MM-DD")
    args = ap.parse_args()

    if args.today:
        try:
            set_today(parse_iso(args.today, "--today"))
        except ValueError as e:
            raise SystemExit(f"error: {e}")

    override = None
    if args.last_frost or args.first_frost:
        if not (args.last_frost and args.first_frost):
            raise SystemExit(
                "--last-frost and --first-frost must be given together.")
        try:
            override = validate_override(args.last_frost, args.first_frost)
        except ValueError as e:
            raise SystemExit(f"error: {e}")

    # Resolve the region before touching the model: a typo should say so
    # immediately rather than after loading 1.7B of weights.
    try:
        resolve(args.region)
    except KeyError as e:
        raise SystemExit(str(e.args[0]))

    try:
        if args.checklist:
            print(render_checklist(args.region, override))
        else:
            print(advise(args.region, override))
    except KeyError as e:
        raise SystemExit(str(e.args[0]))


if __name__ == "__main__":
    main()