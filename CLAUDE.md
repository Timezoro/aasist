# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download ASVspoof 2019 LA dataset
python download_dataset.py

# Train AASIST
python main.py --config ./config/AASIST.conf

# Train AASIST-L (larger variant)
python main.py --config ./config/AASIST-L.conf

# Train baselines
python main.py --config ./config/RawNet2_baseline.conf
python main.py --config ./config/RawGATST_baseline.conf

# Evaluate a pretrained model (--eval flag loads model_path from config)
python main.py --eval --config ./config/AASIST.conf

# Train with custom comment tag (appended to output dir name)
python main.py --config ./config/AASIST.conf --comment my_experiment
```

Results are saved under `./exp_result/{track}_{config_name}_ep{epochs}_bs{batch_size}/`.

## Architecture

**Entry point:** `main.py` handles train / validate / evaluate in a single script.

**Dynamic model loading:** Models are loaded via `importlib.import_module("models.{architecture}")` where `architecture` comes from the config file's `model_config.architecture` field. Every model file must expose a class named `Model`.

**Config files** (`config/*.conf`) are JSON with two top-level sections:
- `model_config` — passed directly to the `Model` constructor (architecture name + hyperparameters)
- `optim_config` — optimizer/scheduler settings consumed by `utils.create_optimizer`

**Data pipeline** (`data_utils.py`):
- `genSpoof_list` parses ASVspoof 2019 protocol `.txt` files into file lists and binary labels (bonafide=1, spoof=0)
- Audio is fixed-length: training uses random crop/tile to 64600 samples; eval uses front-crop/tile (`pad`)
- Dataset root must be set via `database_path` in the config (default `./LA/`)

**Training loop** (`main.py`):
- Loss: weighted `CrossEntropyLoss([0.1, 0.9])` (upweights spoof class)
- Saves best checkpoint by dev EER; optionally evaluates on eval set every time dev EER improves (`eval_all_best`)
- Applies SWA (Stochastic Weight Averaging) across all best-dev-EER epochs at the end
- TensorBoard logs written to the model tag directory

**Evaluation** (`evaluation.py`): computes EER and min t-DCF per ASVspoof 2019 protocol. Requires ASV scores file (`asv_score_path` in config).

**Custom model checklist:**
1. Create `models/YourModel.py` with a `class Model(nn.Module)`
2. `forward(x, Freq_aug=False)` must return `(intermediate, logits)` — only `logits` is used for loss/scoring; `batch_out[:, 1]` is the spoof score
3. Add a config file pointing `architecture` to your filename (without `.py`)
