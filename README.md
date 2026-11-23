# Relational identifiability in a toy world

Question: two languages share no words. A fact is seen only in language A. Can a model
answer it in language B? And does that depend on how well the animal is defined by its
relations?

We build a small artificial world of animals and write it in two made-up languages
(A: 2 CJK characters per word, B: 3 Hiragana per word, no shared tokens). The hearts
facts of some animals ("tested" animals) are only in A. We ask them in B.

k is the number of animals that look exactly the same in B (the animal itself included).
If the model can only match animals across languages through their relations, the best
it can do on a tested fact is 1/k. Chance is 1/8.

## Folders

```
world/     build the worlds
  refine.py    colour refinement: which animals can be told apart (used to check every web and world)
  web.py       food web where no group can be swapped with another
  build.py     one world: facts, sentences in A and B, vocabularies
train/
  train.py     small BERT (4 layers, hidden 128), masked LM, 100 epochs
eval/
  evaluate.py  all metrics for one model (memory, transfer, alignment)
scripts/
  make_worlds.sh   build the web and all worlds
  run_model.sh     train + evaluate one model
  run_k.sh         k experiment (4 worlds x 3 seeds)
  run_ablation.sh  what tells animals apart inside a group (3 worlds x 3 seeds)
  collect.py       results/models/*.json -> results/all_models.csv + summary tables
notes/
  worlds.md    every world we tried, in order, with results
  k.md         the k experiment: what changes with k and what stays the same
results/
  models/          one json per trained model
  all_models.csv   one row per model
worlds/    built worlds and models (not in git, rebuild with make_worlds.sh)
```

## Run

```
zsh scripts/make_worlds.sh
nohup caffeinate -dims zsh scripts/run_k.sh &
nohup caffeinate -dims zsh scripts/run_ablation.sh &
python scripts/collect.py
```

Set `PY=/path/to/python` if `python` is not the right environment.

### Windows with an NVIDIA GPU (native PowerShell)

From the repository root, create a Python environment and install a CUDA-enabled
PyTorch build following the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
For an RTX 50-series GPU, use a current build that supports Blackwell and a recent
NVIDIA driver. For example, with a CUDA 13.0-compatible driver, PyTorch's
official CUDA 13.0 wheel for Python 3.11 can be installed with
`python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130`.
Then install the remaining packages:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install CUDA-enabled torch using the Windows CUDA command from pytorch.org first.
.\.venv\Scripts\python.exe -m pip install "transformers<5" datasets accelerate tokenizers
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Run a short end-to-end check first. It uses only one world/seed and writes results
to `results/smoke/`, without changing the full summary. It checks the pipeline,
not whether the model has learned the facts:

```powershell
.\scripts\run_windows_gpu.ps1 -Ks 1 -Seeds 42 -Jobs 1 -Epochs 1
```

Then run the k experiment, with two training processes on the same GPU:

```powershell
.\scripts\run_windows_gpu.ps1 -Jobs 2
```

`-Jobs` bounds the **number of simultaneous models**, not the batch size. On a
12 GB laptop GPU, start with `-Jobs 2` and change to `-Jobs 1` if CUDA runs out
of memory or concurrency is slower due to contention. `-Jobs 1` preserves the
original sequential execution. The default model settings (100 epochs, batch 64,
FP32) stay the same. `-BuildOnly` creates the worlds; `-SkipBuild` reuses them;
`-DryRun` prints commands. To rerun just k=2 with two seeds:

```powershell
.\scripts\run_windows_gpu.ps1 -Ks 2 -Seeds 42,43 -Jobs 2 -SkipBuild
```

Training logs go to `worlds/nestk_k*__model_*.train.log`; evaluation logs have
`.eval.log`. Full results are saved under `results/models/`, followed by an
update to `results/all_models.csv`. Each training process checks for CUDA;
`eval/evaluate.py` currently evaluates on CPU. The launcher waits for every job
and reports failures instead of regenerating the summary from a partial run.

#### Conda instead of `.venv`

In PowerShell, create and activate a Conda environment, install the Windows CUDA
PyTorch wheel using the command from the selector above, and install the remaining
dependencies:

```powershell
conda create -n toy-gpu python=3.11 -y
conda activate toy-gpu
# Install CUDA-enabled torch using the Windows CUDA command from pytorch.org first.
python -m pip install "transformers<5" datasets accelerate tokenizers
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
.\scripts\run_windows_gpu.ps1 -Ks 1 -Seeds 42 -Jobs 1 -Epochs 1
```

The launcher uses the active Conda environment's `python.exe` automatically.
It also accepts `-Python C:\path\to\env\python.exe` to select one explicitly.

### Independent world seeds and model seeds

The original k experiment changes only the model seed on one fixed world. To
repeat the experiment on three independently generated worlds, with three model
initializations per world and k (36 trained models in total), run from PowerShell:

```powershell
.\scripts\run_windows_gpu.ps1 -WorldSeeds 42,43,44 -Seeds 42,43,44 -Ks 1,2,4,8 -Jobs 1 -Resume
```

`-WorldSeeds` changes the food web, held-out groups, facts and vocabularies;
`-Seeds` changes the model initialization and training randomness. For a 12 GB
GPU, use `-Jobs 1` initially; try `-Jobs 2` only after checking speed and memory.
The default is 100 epochs. `-Resume` skips completed result JSONs after an
interruption. `-BuildOnly` constructs the worlds without training, and
`-SkipBuild` reuses previously constructed worlds.

The new worlds are named `worlds/nestk_w<world-seed>_k<k>` and cannot overwrite
the original `nestk_k<k>` worlds. After every requested model succeeds, the
runner prints a Markdown table like the one below and writes
`results/multiworld_models.csv` (one row per model) and
`results/multiworld_summary.csv` (one row per k). The summary's mean and sample
standard deviation use one mean per world, averaged over its model seeds; with
only one world the SD is shown as zero. The original `results/all_models.csv`
and its historic results stay separate. The k=4 and k=8 worlds are still built
and checked separately for each world seed.

## Main result (32 groups, mean ± SD over 3 seeds)

| k | best possible 1/k | exact alignment | LM head | probe A to B | probe B |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.95 ± 0.04 | 0.305 ± 0.034 | 0.380 ± 0.100 | 0.302 ± 0.032 |
| 2 | 0.50 | 0.48 ± 0.03 | 0.206 ± 0.023 | 0.232 ± 0.052 | 0.206 ± 0.030 |
| 4 | 0.25 | 0.27 ± 0.03 | 0.154 ± 0.016 | 0.193 ± 0.016 | 0.151 ± 0.016 |
| 8 | 0.125 | 0.09 ± 0.03 | 0.112 ± 0.009 | 0.112 ± 0.012 | 0.115 ± 0.012 |

Chance is 0.125. See `notes/k.md` for the setup and `notes/worlds.md` for how we got here.

## Metrics

All from `eval/evaluate.py`. Every forced choice and probe has 8 classes (chance 1/8).

- memory: `a_seen` held-out facts asked in A, `b_other` untested animals asked in B
- transfer (held-out facts asked in B):
  `lm_fc` LM head, choose among the 8 hearts values;
  `lm_free` LM head, top-1 over all words;
  `probe_ab` linear probe trained on A, tested on B;
  `probe_b` linear probe trained on untested animals in B
- alignment: for every animal in B, its nearest animal in A (cosine, each language's mean removed).
  `exact` it is the same animal, `group` same food group, `lookalike` same look-alike set
- values: the same for value words (attributes, hearts, tags)
