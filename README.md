# MSNO: Multi-State Neuron-Level Optimization for Continual Speech Deepfake Detection

Implementation accompanying **MSNO: Multi-State Neuron-Level Optimization for Continual Speech Deepfake Detection**.

MSNO addresses sequential learning for speech deepfake detection using neuron-level state information to guide optimization. The implementation combines class-conditioned neuron states, gradient modulation, historical drift constraints, and cross-task knowledge distillation.

## Model and implementation

The detector uses an XLS-R backbone, temporal mean and standard-deviation pooling, and a two-layer classifier. The XLS-R backbone is frozen by default. Neuron states distinguish shared, bona fide, spoof, and free roles, and guide updates to the trainable classifier.

The main implementation is organized as follows:

```text
.
├── main.py                       # MSNO training and evaluation
├── role.py                       # Neuron states and optimization engine
├── model.py                      # XLS-R backbone, pooling, and classifier
├── UnifiedDataset.py             # Manifest-based audio loading
├── train_baselines.py             # Baseline training and shared utilities
├── cl_baselines.py                # Continual learning baseline strategies
├── build_manifests.py             # Unified manifest generation
├── check_meta_audio_consistency.py # Optional data consistency checks
├── inspect_datasets.py            # Optional dataset structure inspection
├── database/                     # Bundled protocol files
├── fairseq-a540.../               # Bundled fairseq source
├── requirements.txt
└── train.sh                       # MSNO training configuration
```

Some internal names and command-line options retain the earlier `irnr` terminology, including `InteractionRoleEngine`, `--alpha_irnr`, and `--irnr_diagnostics`. Use these names when invoking the current code. `train_baselines.py` is also imported by `main.py` and is required for MSNO training.

## Installation

The commands below use the Python 3.7 / PyTorch 1.12.1 stack documented by the upstream implementation as a starting point. They are not a newly verified environment specification. Use a Linux environment with a compatible CUDA setup and C++ build tools for fairseq.

```bash
conda create -n msno python=3.7
conda activate msno

# Select PyTorch packages compatible with your CUDA environment.
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

# Extract only if the source directory is not already present.
unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip
pip install --editable ./fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1

pip install -r requirements.txt
pip install numpy tqdm soundfile
```

`requirements.txt` is a partial dependency list; PyTorch and the bundled fairseq must be installed separately. The bundled fairseq uses legacy Hydra/OmegaConf dependencies, so preserve its dependency constraints when setting up the environment.

Prepare an XLS-R 300M checkpoint, such as `xlsr2_300m.pt`, and pass its location through `--ssl_checkpoint`. Audio datasets and pretrained weights are not included.

## Data preparation

The project separates original audio datasets from generated manifests. The default raw-data root is `/root/autodl-tmp`; the default manifest root is `/root/autodl-tmp/continual_manifests`.

### Original datasets

`build_manifests.py` expects the following organization. Placeholder filenames illustrate the structure rather than actual sample names.

```text
RAW_ROOT/
├── ASVspoof2019_LA/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.eval.trl.txt
│   ├── train/flac/*.flac
│   └── eval/flac/*.flac
├── for-norm/
│   ├── training/
│   │   ├── real/...
│   │   └── fake/...
│   └── testing/
│       ├── real/...
│       └── fake/...
├── XMAD/
│   └── <language>/
│       ├── train/
│       │   ├── meta.csv             # meta.txt is also accepted
│       │   ├── real/...
│       │   └── fake/...
│       └── test/
│           ├── meta.csv
│           ├── real/...
│           └── fake/...
├── ASVspoof5/
│   ├── ASVspoof5.train.tsv
│   ├── ASVspoof5.eval.track_1.tsv
│   ├── flac_T/*.flac
│   └── flac_E_eval/*.flac
├── release_in_the_wild/
│   ├── meta.csv                    # file and label columns
│   └── <audio paths listed in meta.csv>
├── CodecFake/
│   ├── real/...
│   └── fake/...
├── ASVspoof2021_LA_eval/
│   ├── trial_metadata.txt
│   └── flac/*.flac
├── ASVspoof2021_DF_eval/
│   ├── trial_metadata.txt
│   └── flac/*.flac
├── MLAAD/                          # MLADD is also accepted
│   └── ...                        # Recursively collected as spoof audio
└── M-AILABS/
    └── ...                        # Recursively collected as bona fide audio
```

For ASVspoof5, `Asvspoof5` is also accepted as the directory name. The builder also accepts the training protocol names `ASVspoof5.train.track_2.trial.tsv` and `ASVspoof5.training.track_2.trial.tsv`, and the evaluation protocol name `ASVspoof5.eval.track_2.trial.tsv`.

XMAD metadata must contain an `is_fake` column and a sample-name column, preferably `sample_name`. Samples are located under each language's `train/real`, `train/fake`, `test/real`, or `test/fake` directory. Missing XMAD audio entries are skipped with a warning.

### Generate unified manifests

```bash
python build_manifests.py \
  --raw_root /path/to/raw_datasets \
  --out_root /path/to/continual_manifests \
  --seed 1234
```

This writes CSV manifests without moving or copying audio. By default, all parsed samples are retained. `--max_per_class N` limits each class within each output split; `--skip_large_external` omits the combined MLAAD/M-AILABS test set. Check the printed messages for skipped datasets or parsing errors before training.

```text
continual_manifests/
├── ASVspoof2019_LA/
│   ├── train/meta.csv
│   └── test/meta.csv
├── for-norm/
│   ├── train/meta.csv
│   └── test/meta.csv
├── XMAD/
│   ├── train/meta.csv
│   └── test/meta.csv
├── ASVspoof5/
│   ├── train/meta.csv
│   └── test/meta.csv
├── release_in_the_wild/test/meta.csv
├── CodecFake/test/meta.csv
├── ASVspoof2021_LA_eval/test/meta.csv
├── ASVspoof2021_DF_eval/test/meta.csv
└── MLAAD_M-AILABS/test/meta.csv
```

The generated CSV columns are:

```csv
sample_name,is_fake,audio_path,source,split,language,attack
```

Labels are `0` for bona fide speech and `1` for spoof speech. Generated `audio_path` values are absolute paths into the original datasets. Regenerate or update them if the audio is moved to another machine.

The loader also supports relative paths in `audio_path`, `rel_path`, or `path`. Without these fields, it falls back to `<dataset>/<split>/real/<sample_name>` or `<dataset>/<split>/fake/<sample_name>`. Audio is converted to mono and padded or truncated to 64,600 samples; the loader does not resample it, so prepare a consistent sample rate compatible with the backbone.

## Train MSNO

The configured continual learning sequence is:

```text
ASVspoof2019_LA → for-norm → XMAD → ASVspoof5
```

The five external evaluation datasets are In-the-Wild, CodecFake, ASVspoof2021 LA, ASVspoof2021 DF, and the combined MLAAD/M-AILABS set.

The following command follows the current `train.sh` configuration with portable paths:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --database_path /path/to/continual_manifests \
  --ssl_checkpoint /path/to/xlsr2_300m.pt \
  --train_tasks ASVspoof2019_LA,for-norm,XMAD,ASVspoof5 \
  --external_tasks release_in_the_wild,CodecFake,ASVspoof2021_LA_eval,ASVspoof2021_DF_eval,MLAAD_M-AILABS \
  --start_task_idx 0 \
  --run_dir runs/msno/seed_1234 \
  --batch_size 20 \
  --eval_batch_size 20 \
  --num_epochs_per_task 50 \
  --early_stop_patience 1 \
  --role_activation_tau 1.0 \
  --role_temperature 0.7 \
  --gamma 0.05 \
  --drift_warmup_epochs 2 \
  --eps0 0.1 \
  --penalty_rho 10.0 \
  --lambda_lr 0.5 \
  --lambda_max 20 \
  --projection_margin 2.0 \
  --distill_weight 0.7 \
  --distill_temperature 2.0 \
  --conflict_scale_shared 0.2 \
  --irnr_diagnostics \
  --seed 1234
```

Alternatively, edit the data, checkpoint, and output paths in `train.sh`, then run `bash train.sh`.

The default backbone is frozen. The classifier-only distillation teacher shares frozen XLS-R features; if enabling `--unfreeze_ssl`, set `--distill_weight 0` as required by the current implementation.

To continue from a completed task boundary, provide `--engine_path /path/to/task_N_NAME_best.pt` and set `--start_task_idx N` alongside the same training configuration. This index is zero-based: after task 1, use `--start_task_idx 1` to begin task 2. This restores the saved model and engine; it is not an exact mid-epoch optimizer-state resume.

## Train baselines

Implemented strategy names include `fine_tuning`, `er`, `lwf`, `ewc`, `si`, `mas`, `agem`, `gem`, `rawm`, `rwm`, and `rego`. These are adaptations to the shared detector and training pipeline.

```bash
CUDA_VISIBLE_DEVICES=0 python train_baselines.py \
  --method ewc \
  --database_path /path/to/continual_manifests \
  --ssl_checkpoint /path/to/xlsr2_300m.pt \
  --train_tasks ASVspoof2019_LA,for-norm,XMAD,ASVspoof5 \
  --external_tasks release_in_the_wild,CodecFake,ASVspoof2021_LA_eval,ASVspoof2021_DF_eval,MLAAD_M-AILABS \
  --run_dir runs/baselines/ewc/seed_1234 \
  --seed 1234
```

Pass the external task names explicitly as above: the baseline parser's default currently spells the combined dataset `MLADD_M-AILABS`, whereas the manifest builder produces `MLAAD_M-AILABS`.

## Evaluation and outputs

Evaluation is integrated into training. After each task, the code evaluates the task test sets and, unless disabled, the configured external test sets. The main outputs are:

```text
RUN_DIR/
├── config.json
├── checkpoints/
│   └── task_<number>_<name>_best.pt
├── metrics/
│   ├── train_epochs.csv
│   ├── eer_matrix.csv
│   ├── auc_matrix.csv
│   ├── external_metrics.csv
│   └── final_summary.json
└── scores/
    └── ...
```

EER and AUC are computed by the shared training utilities. The summary includes final average performance and forgetting statistics. Outputs depend on which evaluations are enabled.

- `--skip_external_eval`: skip external datasets.
- `--skip_eval`: skip evaluation.
- `--irnr_diagnostics`: enable optimization diagnostics.
- `--save_mechanism_data`: save additional gradient and neuron-state records.

This repository currently has no separate checkpoint-only evaluation entry point. Do not use the old upstream `--is_eval` / `--eval` commands with the current `main.py`.

For the full argument lists:

```bash
python main.py --help
python train_baselines.py --help
python build_manifests.py --help
```

## Acknowledgments

The project builds on the upstream [SLSforASVspoof-2021-DF](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF) implementation and the bundled fairseq source. Please retain the applicable third-party license and attribution notices when redistributing these components.
