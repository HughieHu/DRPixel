# Counterfactual credit reweighting

This directory prepares the training data for the trained DR Pixel model and trains it. The final
training set is released in the `train/` directory of [`HughieHu/drpixel-bench`](https://huggingface.co/datasets/HughieHu/drpixel-bench#training-data-train), so you can
train the model directly from it. The same directory holds the trajectories with their raw credits
under `credit/`, from which the scripts here rebuild that training set. The trained model is
[`HughieHu/DR-Pixel-Trained`](https://huggingface.co/HughieHu/DR-Pixel-Trained), and the benchmark
and evaluation harness are in [`../eval`](../eval).

## Why reweight

We train DR Pixel with supervised fine-tuning on trajectories sampled from GPT-5.4. These
trajectories often contain repeated or low-value searches that do not contribute to the final
answer. Standard SFT imitates them along with everything else, which teaches the student to search
too much and to spend its retrieval budget poorly. Counterfactual credit reweighting keeps the
simplicity of SFT but weights each step by its contribution to the final answer.

## The method

For a teacher trajectory with gold answer $y^\star$, let $H_i$ be the history after $i$ searches
and $\pi_0$ the frozen base model. How strongly a history supports the gold answer is measured by
$\ell(H) = \log \pi_0(y^\star \mid H)$. Reasoning steps and search steps get separate credit:

$$c^\tau_i = \ell(H_T) - \ell(H_T \setminus \tau_i), \qquad c^a_i = \ell(H_i) - \ell(H_{i-1}).$$

The credit of a reasoning step $\tau_i$ is how much support drops when it is removed from the full
trajectory. The credit of a search step $a_i$ is how much support its results add.

Raw credits are unbounded, so each step type is converted to weights by rank. A step with credit
$\le 0$ gets weight 0. A positive credit that ranks $k$-th among $n$ positive credits of its type
gets weight $2k/n$, so positive weights average about 1, the scale of standard SFT.

Each reasoning step produces the search query that follows it, so the two weights are coupled. The
likelihood gain of a reasoning step is how much it raises the likelihood of its action, compared
with an empty reasoning block. A reasoning step is masked if its action was masked or if its gain is
below the median over all reasoning steps. Otherwise it keeps its weight, and a step whose credit
had given it weight 0 is restored with the median positive reasoning weight. Final-answer tokens get
a constant weight $\lambda = 1.5$.

The trainer multiplies each message's token loss by its weight, so the training step itself is
ordinary SFT, and the whole method lives in data preparation.

## Training data

The `train/` directory of [`HughieHu/drpixel-bench`](https://huggingface.co/datasets/HughieHu/drpixel-bench#training-data-train) holds the 2,496 GPT-5.4 trajectories the
paper's model was trained on, with their per-step weights, as `train.jsonl`, plus the page images
they reference as tar shards. Image paths in `train.jsonl` are relative to `train/`, so extract the
shards there and train from that directory:

```bash
hf download HughieHu/drpixel-bench --repo-type dataset --include "train/*" --local-dir drpixel
cd drpixel/train && for t in image-shards/*.tar; do tar -xf "$t"; done     # creates images/
python /path/to/DRPixel/train/register_stepw.py
BASE=/path/to/Qwen3.6-27B DATA=train.jsonl OUT=/path/to/out /path/to/DRPixel/train/train.sh
```

## Pipeline

Five scripts, run in order. Steps 0 and 3 run the frozen base model and need a GPU; the others are
plain Python.

| step | script | what it does |
|---|---|---|
| 0 | `compute_credit.py` | scores every search step and stores its credits on the step as `cf_query` (search) and `cf_think` (reasoning) (needs a GPU) |
| 1 | `make_weights.py` | turns the credits into rank-based weights, separately for reasoning and queries |
| 2 | `split_think.py` | splits each assistant message at `</think>`, so the reasoning and the query can carry different weights |
| 3 | `compute_nextact.py` | computes each reasoning step's likelihood gain on the action that follows it (needs a GPU) |
| 4 | `build_train.py` | masks or keeps each reasoning step by that gain and writes the training file |

```bash
python compute_credit.py --data trajectories.jsonl --out credit.jsonl --model /path/to/base-model    # GPU; --shard/--nshard to parallelise
python make_weights.py   --shards 'credit.shard*.jsonl' --out weighted.jsonl
python split_think.py    weighted.jsonl split.jsonl
python compute_nextact.py --data split.jsonl --out nextact.jsonl --model /path/to/base-model   # GPU; --shard/--nshard to parallelise
python build_train.py    --split split.jsonl --na 'nextact.shard*.jsonl' --out train.jsonl
```

`split_think.py --uniform` writes the same data with uniform weights, which is the vanilla SFT
control.

Step 0 scores the trajectory's own final answer, the text inside its last `<answer>`, which is why
only trajectories with a correct answer are used. Each trajectory takes 2 + (number of searches) +
(number of reasoning steps) forward passes, the reasoning ones at full length. The model needs
about 125–142 GB in bf16 at full trajectory length, so one H200 or two 80 GB GPUs; on an H200 a
trajectory takes about a minute.

### Rebuilding our training set

`train/credit/trajectories.jsonl` in the dataset is the input to step 1 for the paper's model: the
2,496 trajectories with their credits (`cf_query`, `cf_think`) and step-3 gains (`na_think`), so
steps 1, 2 and 4 run without a GPU. Run them in the dataset's `train/` directory with the images
extracted:

```bash
python make_weights.py --shards credit/trajectories.jsonl --rank_pool credit/rank_pool.json --out weighted.jsonl
python split_think.py  weighted.jsonl split.jsonl
python build_train.py  --split split.jsonl --out train.jsonl
```

## Data format

The files are JSONL with one chat-format trajectory per line: `messages` (role and content; page
images appear as `<image>` in the content) and `images` (image paths in order of appearance). Step 0
needs the final answer in the last message, inside `<answer>`; step 1 expects every search-step
assistant message to carry `cf_query`, and `cf_think` when it has reasoning. Each later step adds or
rewrites a per-message `loss_scale`.

The training file must contain only `messages` and `images`: ms-swift fails to load a dataset with
extra columns.

## Training

The weights are stored in the data, as a numeric `loss_scale` on every assistant message. ms-swift
needs a two-line plugin to use these values as they are, instead of rounding them to 0 or 1:

```python
class StepWeightLossScale(LossScale):
    is_binary = False
```

`register_stepw.py` installs it into ms-swift under the name `stepw`. Then:

```bash
python register_stepw.py                                       # once per environment
BASE=/path/to/Qwen3.6-27B DATA=train.jsonl OUT=/path/to/out ./train.sh
```

`train.sh` is the recipe used in the paper: Qwen3.6-27B with LoRA (rank 32, alpha 64, dropout 0.05)
on all linear layers, one epoch at learning rate 1e-4 with an effective batch size of 16, maximum
length 131,072 tokens, padding-free batching, Liger kernels, bf16 and FlashAttention, with
`--loss_scale stepw --disable_ignore_empty_think true`. The model is trained directly from the base
model, with no warm-up stage. Answer turns keep the weight the input trajectories carry
($\lambda = 1.5$ in ours); the pipeline only reweights the reasoning and queries of search steps.

`train.sh` sets `MAX_PIXELS=602112`, which caps page images at that size during training; at full
resolution an H200 runs out of memory. Evaluation serves pages at full resolution.

## Check that the weights applied

ms-swift's default loss scale is binary and turns every weight into 0 or 1. If the plugin is not in
effect, the run silently becomes plain SFT and nothing downstream looks wrong, and
`--loss_scale stepw` resolving without an error does not rule this out.

The reliable check is the logged loss. Train a control on the same file with
`split_think.py --uniform` and compare the two curves. If the weights reach the trainer, the losses
differ from the first logged step; identical curves mean they do not.

## Results

Table 15 of the paper adds the three ingredients (the answer weight, the counterfactual credit and
the coupling between reasoning and search) to vanilla SFT one at a time, on the same 2,496
trajectories with the same recipe. Each one reduces the number of searches per question.

## Citation

See the [top-level README](../README.md#citation).
