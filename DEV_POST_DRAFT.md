# DRAFT — not published

Submission draft for the Hacktoberfest 2026 DEV Open-Source AI Challenge,
Week 1: "Touch Grass."

Required tags: `#devchallenge` `#hf26challenge`

---

## Title

**I Built a Garden Planner That Runs a 1.7B Model Entirely Offline — Because It Was
Supposed to Make Me Go Outside**

## Body

Last October I stood in my yard holding a seedling I had no business planting. The
frost map on my phone said it was fine. The map was wrong, or the date was stale, or
I'd just forgotten. Either way I had wasted a season on a bad lookup.

So I built the lookup I wanted. `.venv/bin/python garden.py iowa`

### The problem with asking a model "what should I plant?"

Small models are confidently wrong at date arithmetic. Not occasionally — routinely.
Ask a 1.7B model "will these 65-day tomatoes finish before first frost in Texas?"
and it will invent a number, round it, and sound certain while doing it. For a
question where a wrong answer costs you a whole season, that is disqualifying.

So I split the task, and the split is the whole design:

| Step | Who does it | Why |
|---|---|---|
| Frost dates, planting windows, "will this finish before first frost" | `frost.py`, plain Python date math | A wrong frost date silently corrupts every recommendation built on it. |
| Explaining the result to a human, adapting it to their situation | local Qwen3-1.7B, Apache-2.0 | This is genuinely a language task, and it's where a small model is actually fine. |

The model is handed **finished, verified facts** and asked to write three sentences
about them. It is never asked a question it could get wrong.

That constraint is what makes a model small enough for a 4 GB laptop genuinely
useful here instead of a novelty.

### It runs with the wifi off

No API key. No account. No server. No network call at query time. The model is
Qwen3-1.7B (Apache-2.0 open weights) running locally through `llama-cpp-python`.
Once you have downloaded it, the whole thing works in airplane mode.

That is partly a privacy position — I'm not shipping my location and my garden plan
to a third party — but mostly a *trust* one. When the answer is computed locally
from a table I can read, I can check it. When it comes from a hosted endpoint, I'm
trusting that the provider computed it correctly, and I can't.

### The part I did not expect: the tests found real bugs

The interesting part of this project isn't the model call, it's what happened when I
told it what today's date actually was.

**Bug one, pinned by a test.** When the fall frost date had already passed, the code
advanced *fall* to next year but left *spring* in the past. Iowa got a 518-day
growing season. Frost dates only mean anything as a pair belonging to the same
season; when the season closes, both have to roll forward together.

**Bug two, found by a test failing for the right reason.** The original code asked
"could this crop finish?" But the question a gardener has in October is "can I still
*plant* it?" Those are different questions. A 65-day tomato crop can finish in Texas
in October — while the window to sow it closed back in May. The old function
happily recommended planting tomatoes with no realistic chance of ripening. There are
now two functions, because there are two questions.

Neither bug was subtle to find once the clock was pinnable. Both were invisible
without it.

### Checking the model isn't just guessing

"Does it run?" is a weak test. A small model will always return *something*, and a
tool that silently falls back to a canned string when the model produces nothing
will happily pass every test you write while no AI is involved at all.

So `test_model_audit.py` runs the real GGUF and checks every number and every plant
name in the output against the Python facts the model was given. If the model says
"wait 360 days" and it was given 206, that's a failure. If it names a crop that
wasn't in its list, that's a failure. And if the output turns out to be the
deterministic fallback, the test fails explicitly — because a fallback is free of
hallucinations by construction, and would otherwise pass in silence.

It audits all 40 regions, twice: once at today's date, once pinned to a spring date
so the in-season prompt path gets the same scrutiny as the off-season one.

### Getting people outside, specifically

The default output is a paragraph. `--checklist` prints one page sized for a phone
held in a garden bed:

```
Garden checklist - iowa - 2026-10-11

PLANT (3)
- [ ] garlic
- [ ] cover crop
- [ ] compost for beds

CHECK (1)
- [ ] soil: dig 5cm down and squeeze -- is it crumbly, not smeared?

LOOK (1)
- [ ] find one pollinator on a flower, or one pest on a leaf

Nothing outdoors until 2027-05-05.
```

That last line is the actual point of the project. There are no 40 states where the
answer to "what should I plant right now" is always "something to plant." A
gardener in Iowa in October has real work to do — it's just not sowing. The tool
tells you the season reopens on a date, gives you the jobs that matter this month,
and sends you back outside with one thing to look at.

### Why open innovation matters here

Three reasons, concretely:

**The weights are the part you can inspect.** Qwen3-1.7B is Apache-2.0. I can read
what it is, run it on my own hardware, and know it will still work in ten years
because it is a file on my disk. A hosted model is a rental that can be deprecated,
repriced, or silently changed underneath me. For an agronomy tool where I am being
told when to put a seed in the ground, "it is just a file" is a feature.

**Small and local is a legitimate engineering choice, not a compromise.** The split
design exists *because* the model is 1.7B. A weak model that is only asked what it's
good at is more useful than a strong model it can phone home to. Someone without a
GPU, without a subscription, or without a reliable connection gets the same tool.

**The part that most deserves scrutiny should not be a black box.** The safety-critical
logic here — frost dates, maturity windows, "will this finish" — is deterministic
Python you can read line by line. The model only does the part where being wrong is
cheap: turning correct numbers into sentences a person will act on. I think that's
the general shape responsible AI work should take. Put the verifiable part in code
where it can be checked, and use the model for the part that needs a voice.

### Honest limitations

- Frost dates are approximate regional averages following commonly published 50%-
  probability maps. A starting point, not a forecast. A single cold night in your
  own yard beats any table. `--last-frost` / `--first-frost` override them.
- US states only, northern hemisphere only.
- A 1.7B model is a 1.7B model. The audit test is there to keep it honest rather
  than to pretend it doesn't need watching.
- Verified on an RTX 2050 (4 GB VRAM), 16 GB RAM, 12-core CPU, CachyOS/Linux —
  but the `llama-cpp-python` build used here is **CPU-only**
  (`llama_supports_gpu_offload()` returns `False`), so the GPU sits idle and each
  answer takes about **10 seconds** on the CPU. The code offloads whatever the
  build supports, so a CUDA build would use the GPU with no source change.

### Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -C - -o models/Qwen3-1.7B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf

.venv/bin/python test_frost.py       # fast
.venv/bin/python test_adversarial.py # fast
.venv/bin/python test_model_audit.py # slow, needs the model
.venv/bin/python garden.py iowa
.venv/bin/python garden.py iowa --checklist
```

Then turn off your wifi and try it again.

---

## To-do before publishing

- [ ] Attach a terminal recording of `garden.py iowa` (real model output)
- [ ] Attach a screenshot/GIF of `--checklist` on a phone
- [ ] Add repo link if the source is published
- [ ] Confirm both tags `#devchallenge` and `#hf26challenge` are present
- [ ] Replace "40 states" phrasing with the accurate "40 regions"