# Evaluating on DRPixelBench

DRPixelBench has 500 questions written from scientific papers. Each question is grounded in a
specific page, often in a table, figure or equation, and has a short answer that can be checked
against the gold. To answer it, an agent searches a corpus of 99,055 papers (1,062,153 pages) over
several rounds of retrieval and reasoning.

The harness, the `drbench` package, needs two things from you: a model endpoint, set with three
environment variables, and a search endpoint at `SEARCH_URL` (§3).

```bash
pip install -e '.[client]'
cp .env.example .env                  # your model endpoint, the search endpoint and the judge key
python -m drbench.doctor              # about 20 s: checks both endpoints and whether your model sees images
python -m drbench.eval --config Page
```

## 1. Install

```bash
pip install -e '.[client]'    # from this directory
```

Install in editable mode, because the runner reads the questions from `bench/` in this directory.
The client only uses the Python standard library. The `client` extra adds `certifi`, which avoids
TLS errors on machines with an outdated CA bundle (common on macOS).

## 2. Point it at your model

Set three environment variables, or put them in `.env`, which is loaded automatically. Any endpoint
that speaks the OpenAI chat-completions API works:

```bash
MODEL_NAME=gpt-4o
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_API_KEY=sk-...
```

A local server only needs a different URL:

```bash
# vLLM, SGLang, llama.cpp, LM Studio, ...
MODEL_NAME=Qwen/Qwen3.6-27B
MODEL_BASE_URL=http://localhost:8000/v1
MODEL_API_KEY=EMPTY
```

If your model does not speak this API, subclass `Client` and override `complete()` in
[drbench/model.py](drbench/model.py), the only code that talks to the model:

```python
from drbench import eval, model

class MyClient(model.Client):
    def __init__(self):
        super().__init__(model="my-model", base_url="unused")   # the base class requires both

    def complete(self, msgs, max_tokens):
        # msgs: [{"role": "system" | "user" | "assistant", "parts": [part, ...]}], where a part is
        # {"_t": "text", "text": str} or {"_t": "img", "data": bytes, "mime": "image/png"}
        text, n_in, n_out = my_model(msgs, max_tokens)
        model._add(n_in, n_out, 0)    # optional: per-question token counts
        return text

model._client = MyClient()
eval.main(["--config", "Page"])
```

Before a full run, `python -m drbench.doctor [Text|Page|Text_Fig]` (default `Page`) checks the
model, the search endpoint and the judge, and whether the model actually receives images.

## 3. Retrieval

The client sends queries to a search endpoint at `SEARCH_URL`. It does not need the index, the PDFs
or a GPU itself.

### Hosted endpoint for `Page`

We host the `Page` channel, so a `Page` run needs no download and no GPU on your side:

```bash
export SEARCH_URL=https://drpixel--drpixel-search-gate.us-east.modal.run
```

It serves the retriever of `server/search_server.py` on a GPU and scales to zero when idle, so the
first search after a quiet period waits about a minute. GPU time is limited per day across all
users; once it is used up, searches return HTTP 429 until 00:00 UTC. Pages with nearly tied scores
can come back in a different order than in the paper's runs, because the scores depend slightly on
the GPU. `Text` and `Text_Fig` need your own server (below).

### Data sources

Everything the search side needs is in one HuggingFace dataset,
[`HughieHu/drpixel-bench`](https://huggingface.co/datasets/HughieHu/drpixel-bench): the questions,
the paper list, the OCR text and every indexed page as a PNG (`corpus/`), and both embedding
indices (`index/`).

| artifact | size | used for |
|---|---|---|
| `corpus/papers.jsonl` | 8.9 MB | not needed to run; one line per paper with its source, arXiv id and page count, which tells you which paper a hit comes from |
| `index/page/` | 8.7 GB | `Page` |
| `index/text/` | 36 GB | `Text`, and the text tool of `Text_Fig`. 3,624,786 chunks embedded with Qwen3-Embedding-8B; this is the index used in the paper |
| `corpus/ocr/part-*.jsonl.gz` | 3.6 GB | the OCR text of all 99,055 papers, one JSON object per paper (`doc_id`, `title`, `text`, `pages`). Chunking `text` the same way (strip front matter, collapse blank lines, split on whitespace, cut every 256 words) reproduces the `Text` passages exactly |
| `corpus/pages/` | 473 GB | serving `Page` without PDFs: one PNG per indexed page, plus an index from embedding row to image |
| figure-crop index | | the figure tool of `Text_Fig`; not released yet |
| source PDFs | 393 GB | not redistributed; `corpus/papers.jsonl` gives an arXiv id for 76.3% of the corpus |

Our server renders each page from its PDF at request time with PyMuPDF, at 150 dpi and without a
pixel cap. The images in `corpus/pages/` are the same render (we compared sampled pages byte for
byte), so a server that reads them does not need the PDFs.

### The search server

`server/search_server.py` serves the `Page` and `Text` channels from a local copy of the dataset,
with the same query encoders and scoring as the paper's runs, and defines the HTTP protocol the
client speaks. Download only the parts your configs use:

```bash
pip install -e '.[server]'
hf download HughieHu/drpixel-bench --repo-type dataset --local-dir data \
  --include "index/page/*" --include "corpus/pages/*"     # Page: 482 GB
hf download HughieHu/drpixel-bench --repo-type dataset --local-dir data \
  --include "index/text/*"                                # Text: 36 GB
DATA_DIR=data python server/search_server.py              # then SEARCH_URL=http://localhost:19001
```

The server needs a GPU for the query encoders, Qwen3-VL-Embedding-8B for `Page` and
Qwen3-Embedding-8B for `Text`. If one GPU cannot hold both encoders and both indices, set
`PAGE_DEVICE` and `TEXT_DEVICE` to put the two channels on different devices, or set
`SERVER_PARADIGMS=page` to serve one channel. The text index is kept on the GPU when it fits and in
CPU memory otherwise; `TEXT_INDEX_DEVICE=cuda|cpu` forces the choice. `RETRIEVER_PORT` sets the
port (default 19001). The figure channel of `Text_Fig` is not served yet,
because its index is not released. `/health` lists the channels that are being served
and any that failed to load, and the runner will not start if a channel your config needs is
missing, rather than returning empty results for every question. Images travel base64-encoded in
`image_b64` and the client decodes them back to bytes, so your model sees exactly the bytes it would
see if retrieval ran in the same process.

### Your own PDFs

`server/index_pdfs.py` builds a `Page` index over your own PDFs, in the same layout, so the server
and the agent work on them unchanged. Each page is rendered at 150 dpi and embedded with
Qwen3-VL-Embedding-8B as the benchmark's pages were (on a GPU; `pip install vllm` for the default
backend, or `--backend transformers`). A paper's doc id is its file name; only the pages up to the
references are indexed unless you pass `--pages all`.

```bash
python server/index_pdfs.py --pdfs my_papers/ --out my_data
DATA_DIR=my_data SERVER_PARADIGMS=page PAPER_COMPAT=0 python server/search_server.py
python -m drbench.ask "your question"          # see 4. Run
```

If the server runs on another machine, plan for the bandwidth. A page is 300–480 KB of PNG and
`search_page` returns three, so one search moves 1–2 MB once base64-encoded. At roughly nine
searches per question, a full `Page` run moves several gigabytes. Do not downscale the images or
lower the render resolution to save traffic, because resolution changes the scores.

## 4. Run

| `--config` | what the agent can search | needs vision |
|---|---|---|
| `Page` | `search_page`: whole-page images, top 3 | yes |
| `Text` | `search_text`: 256-word OCR passages, top 5 | no |
| `Text_Fig` | `search_text` and `search_figure`: passages and cropped figures | yes |

`Page` is the DR Pixel setting and the main one in the paper; `Text` and `Text_Fig` are the
OCR-based baselines. `Text` sends no images, so a text-only model can run it. Each passage is
returned whole up to 2,200 characters, just above the 90th percentile of chunk length; the server
applies this cap. Config names are case-insensitive.

```bash
python -m drbench.eval --config Page --limit 10     # try 10 questions first; grading costs money
python -m drbench.eval --config Page                # all 500; rerun the same command to resume
python -m drbench.eval --config Page --no-grade     # retrieval metrics only, no judge calls
python -m drbench.eval --config Page --grade-only   # grade an existing results file
```

To ask a single question of your own, with the same model and search endpoint:

```bash
python -m drbench.ask "Which optimizer does the paper use for pretraining?" [--config Page] [--json]
```

## 5. What gets reported

The runner prints the following, overall and for each evidence type (text, table, figure,
equation):

| metric | meaning |
|---|---|
| Acc | accuracy, as graded by the judge |
| Tool/q | searches per question; the final answer is not counted |
| PgHit | share of questions for which the gold page was retrieved at least once |
| DocHit | the same for the gold paper |
| Acc\|Pg | accuracy on the questions whose gold page was retrieved |
| tokens | input and output tokens per question |

Every record, with the answer, each search and its hits, and per-step token counts, is written to
`results/`.

Keep these conventions in mind when comparing numbers:

- Every question is grounded in exactly one gold page, so under `Page` PgHit is taken over all
  500 questions. Under `Text_Fig` only a figure search can return a page, so PgHit is printed over
  the questions on which the agent searched figures, and over all 500.
- Acc|Pg is conditioned on the gold page. Conditioning on the gold paper instead gives a different
  denominator and a different number.
- Under `Text`, PgHit is shown as n/a rather than 0%, because a text chunk carries a chunk index,
  not a page number.

## 6. Grading

Gold answers are short sentences with specific values, so answers are graded by an LLM judge, as in
the paper: `gpt-5.4-mini` with a strict prompt, taking a majority vote over three judgments. The
rubric is in [drbench/judge.py](drbench/judge.py). It is strict about specificity: an answer that
points in the right direction but leaves out the numbers in the gold answer is marked wrong. If all
three judgments fail for a question, that question is left out of Acc rather than counted wrong.

The judge's model, number of votes, rubric and request format are fixed in the code so that results
stay comparable. You only set `JUDGE_API_KEY` and `JUDGE_BASE_URL`.

## 7. Reproducibility notes

- The system prompt in [drbench/harness.py](drbench/harness.py) is byte-identical to the one used
  for every number in the paper, and a test checks this. As in the paper, models whose name starts
  with `gpt-` get it without the opening "You are a research assistant" persona. Changing the
  prompt makes your results incomparable.
- Every model uses the same in-band `<call_tool>` protocol rather than a provider's native function
  calling, so all providers see the same interface. The paper's GPT-5.4 runs used native tool
  calling, so a GPT-5.4 number from this harness may differ from the paper's.
- Set `CTX_LIMIT` to your model's real context window (default 131072). When the estimated prompt
  passes 75% of it, the oldest page images are dropped. If it is set too low, the model loses
  evidence it could have kept; if it is set too high, the endpoint rejects requests partway through
  an episode.
- At most 24 images stay in context; older ones are replaced by a short note. Each question gets up
  to 20 searches. Both limits are fixed, because every reported run used them.
- Token counts come from the endpoint's own `usage` field. A record's `in_tok` is summed over every
  request in the episode. Each ReAct turn resends the whole history, so this total grows roughly
  quadratically with the number of turns and is much larger than any single prompt.
- If the model still calls a tool on the forced final turn, the tool name is recorded as its answer.
  The paper's runs behaved this way, and we keep it for comparability.

## 8. Layout and data format

```
drbench/
  model.py      the model adapter, the file you change
  harness.py    the agent loop and the system prompt
  retrieve.py   search client
  judge.py      grading rubric and metrics
  eval.py       the runner
  doctor.py     preflight checks
  configs.py    the three configs
server/
  search_server.py      the search endpoint
  retriever.py          page and passage search over the released data
  qwen3_vl_embedding.py the Qwen3-VL-Embedding query encoder (vendored)
bench/
  eval500.json  the 500 questions
.env.example    the settings, documented
```

Runs are written to `results/`, which git ignores.

`bench/eval500.json` has one entry per question:

```
qid           opaque identifier
question      the question
gold_answer   the reference answer
modality      evidence type: text | table | figure | equation     216 / 135 / 102 / 47
doc           the source paper's id, the id that retrieval hits carry
arxiv         its arXiv id, or a corpus id where none is known
page          0-based index of the PDF page holding the evidence
```

`page` uses the same numbering as a hit's `page_idx`, so the two can be compared directly. The paper
calls questions with table, figure or equation evidence Visual questions and the rest Text
questions.
