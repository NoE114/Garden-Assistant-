# Garden planner — Touch Grass, Hacktoberfest Week 1

> "What should I plant this week, around here?"

Runs an open-weight model **entirely on your machine**. No API key, no account,
no server, no network call at query time. Once the model is downloaded, the
whole thing works on a laptop in airplane mode.

```bash
.venv/bin/python garden.py iowa
```

## Why it works offline

The design splits the task in two, because the split is what makes a 1.7B model
usable for something that has to be *correct*:

| Step | Who does it | Why |
|---|---|---|
| Frost dates, planting windows, "will this finish before first frost" | `frost.py`, plain Python date math | A wrong frost date silently corrupts every recommendation built on it. Small models are bad at date arithmetic and confidently wrong. |
| Explaining the result to a human, adapting it to their situation | local Qwen3-1.7B (open weights, llama.cpp) | This is genuinely a language task, and it's where a small model is actually fine. |

The model never sees a question it could get wrong. It is handed finished
facts and asked to write three sentences about them. That is why a model small
enough to run on a 4GB laptop is enough here.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -L -C - -o models/Qwen3-1.7B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
```

`-C -` resumes an interrupted download instead of restarting it, and never
overwrites a complete file with a shorter one. The file must be exactly
1,834,426,016 bytes; `garden.py` checks this and tells you if it is short.

Then:

```bash
.venv/bin/python test_frost.py    # self-check
.venv/bin/python garden.py texas
```

## The interesting bug

`test_frost.py` exists because the first version of this was wrong in a way
that only showed up in October. When the fall frost date had already passed,
the code advanced *fall* to next year but left *spring* in the past — producing
a 518-day "growing season" for Iowa. Frost dates only mean anything as a pair
belonging to the same season; when the season closes, both roll forward
together. That regression is now pinned by a test.

The second fix came from a test failing for the right reason: `grow_window()`
asked "could this crop finish?" but the question a gardener has in October is
"can I still plant it?". A 65-day tomato crop can *finish* in Texas in October
while the window to sow it closed in May. Those are different questions, so
there are now two functions.

## Data honesty

Frost dates are approximate, following commonly published 50%-probability
spring/fall frost maps. They are a starting point, not a forecast. A single
cold night in your own yard beats any table. 40 regions, one table, no API.

## Files

- `frost.py` — frost-date table, planting-window math. No model, no deps.
- `garden.py` — local model call, prompt, CLI.
- `test_frost.py` — self-check, run it directly.
- `test_adversarial.py` — 17 probe functions: dates, invariants, every region.
- `test_model_audit.py` — loads the GGUF and checks the model's output against
  the Python facts. Slow (~1–2 min). Fails if the output is the deterministic
  fallback, i.e. if no inference actually ran.

## Running the tests

```bash
.venv/bin/python test_frost.py          # fast, no model needed
.venv/bin/python test_adversarial.py    # fast, no model needed
.venv/bin/python test_model_audit.py    # slow, needs the model downloaded
```

## Requirements and limitations

- **Hardware**: verified on an RTX 2050 (4 GB VRAM), 16 GB RAM, 12-core CPU,
  CachyOS/Linux. About **10 seconds** per answer. Note that the `llama-cpp-python`
  wheel/source build used here is **CPU-only** (`llama_supports_gpu_offload()`
  returns False), so inference runs on the CPU and the GPU is not used. The
  model is small enough that this is fast enough to be practical.
- **Build**: `llama-cpp-python` has no wheels for Python 3.14, so it compiles
  from source. You need `gcc` and `cmake`. Rebuilding with CUDA support is
  possible but not required, and not what these timings reflect.
- **Internet**: needed once, to download the model and install the one
  dependency. After that, `garden.py` makes no network calls.
- **Accuracy**: frost dates are approximate regional averages, not a forecast
  for your specific yard. Use `--last-frost` / `--first-frost` to override them.
- **Regions**: US states only, and northern hemisphere only.

## License

The model, Qwen3-1.7B, is Apache-2.0 licensed open weights from Alibaba, via
the official `Qwen/Qwen3-1.7B-GGUF` repository. `llama-cpp-python` is MIT.

## Stack

Python 3.14 · llama-cpp-python · Qwen3-1.7B Q8_0 (Apache-2.0 open weights)