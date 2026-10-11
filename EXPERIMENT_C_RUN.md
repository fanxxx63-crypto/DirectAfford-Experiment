# Experiment C: strict LASO runner

`src/run_experiment_c.py` provides training, validation and evaluation for the
unchanged `DirectAffordModel` in `src/model.py`. It fixes the LASO training
entry's omission of the regression head from the optimizer and directly reads
the official LASO files. Existing entry points remain available.

This is a runnable protocol, not a claim of a reproduced paper score.

## Method and protocol

- Original geometry encoder: 256 patches, 32 neighbors, dimension 384, four
  Transformer layers. Geometry and regression head are trained from scratch.
- Frozen `openai/clip-vit-base-patch32` text encoder, normalized pooler output;
  logits are exactly ten times cosine similarity.
- Soft-label BCE averaged over all points; AdamW over **all trainable
  geometry and head parameters**, learning rate 0.0001, weight decay 0.0001.
- Defaults: 300 epochs, batch size 32, four loader workers, FP32. No scheduler,
  early stopping, ranking loss, EMA or learned temperature is added.
- Every epoch uses shuffled training data with `drop_last=True` (503 batches,
  16,096 of 16,120 samples per epoch at batch size 32). Full validation has
  1,215 samples. The highest validation AUC selects `best_auc.pth`; a tie keeps
  the earlier checkpoint. The paired MAE comes from that same checkpoint.
- Default `--query-mode affordance` preserves the original class-name input.
  `canonical` uses official `Question0`; `mixed` uses Question0 with probability
  0.5 and a uniformly selected available rephrasing otherwise during training,
  and Question0 for validation/test. These are explicit protocol variants,
  not interchangeable reproductions. Record the chosen mode before training.
- Coordinates remain unchanged by default, matching the original LASO entry.
  `--normalize` explicitly changes the protocol to centered unit-radius points.
- Evaluation fixes the FPS random stream by seed and restores training RNG
  afterwards. Keep evaluation batch size identical to training because FPS
  randomness is sensitive to batching.

## Dependencies and data

Use Python 3.10 and a CUDA-capable GPU. Install PyTorch from its official CUDA
index first, followed by the pinned companion requirements:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1+cu118
python -m pip install -r requirements-experiment-c.txt
```

Follow the LASO links in `README.md` to obtain the official dataset. The
`--data-root` directory must directly contain `anno_train.pkl`, `anno_val.pkl`,
`anno_test.pkl`, corresponding `objects_*.pkl` files, and
`Affordance-Question.csv`. The reader requires 16,120/1,215/2,416 annotations
and aligned 2,048-point XYZ/mask arrays. Missing or invalid data fails loudly;
it is never replaced by zero geometry, synthetic masks or skipped samples.

Pass `--text-model` a local, fixed official CLIP snapshot, including tokenizer
files and weights. A text-only snapshot saved with `CLIPTextModel` and
`CLIPTokenizer` from the same official revision is also supported. The runner
records hashes for model assets, dataset files and source code. Downloading
OpenShape/Point-BERT weights is unnecessary for this model.

## Commands

The following example explicitly chooses full situational questions. Change
the mode to `affordance` to preserve the original category-name input.
All output paths should be outside the repository; do not commit datasets,
downloaded assets or generated results.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
DATA=/absolute/path/to/LASO_dataset
CLIP=/absolute/path/to/local_clip_snapshot
RUN=/absolute/path/to/new_seed42_run

python src/run_experiment_c.py --phase preflight \
  --data-root "$DATA" --text-model "$CLIP" --query-mode mixed

python src/run_experiment_c.py --phase train \
  --data-root "$DATA" --text-model "$CLIP" --query-mode mixed \
  --seed 42 --epochs 300 --batch-size 32 --workers 4 --output-dir "$RUN"

# After all planned epochs finish, evaluate the validation-selected checkpoint.
python src/run_experiment_c.py --phase evaluate \
  --data-root "$DATA" --text-model "$CLIP" --query-mode mixed \
  --seed 42 --batch-size 32 --workers 4 \
  --checkpoint "$RUN/best_auc.pth" --split test --allow-test --save-scores \
  --output-dir "$RUN/test"
```

Use distinct run directories for seeds 42, 43 and 44. Do not select a seed or
checkpoint by its test score. Report the arithmetic mean for a three-seed
experiment and retain every seed's result.

For a smoke run, add `--limit-train 96 --limit-val 64 --epochs 2` and choose a
separate output directory. `--stop-after-epoch 1` is available only for debug
subsets. Resume by repeating the identical planned training configuration,
omitting the stop flag and adding `--resume "$RUN/last.pth"`.

## Recovery and evaluation guarantees

`last.pth` contains model, optimizer, RNG states, loader RNG, epoch history,
configuration and provenance. Writes are atomic. Recovery resumes at the next
complete epoch; work since the last completed epoch is replayed. The resume
command rejects changed configuration, data, CLIP assets or implementation.
Only load checkpoints from a trusted source.

The final test command requires completed training, the recorded best
checkpoint, all 2,416 test samples and zero skipped AUC values. Debug-trained
checkpoints cannot produce a formal test result. Existing evaluation outputs
are not overwritten.

AUC is the arithmetic mean of per-sample ROC AUC on raw cosine scores, with
binary targets `soft_mask >= 0.5`. MAE compares `sigmoid(10 * cosine)` against
the continuous masks, averaged over points and samples. There is no min-max
rescaling or calibration. Outputs include metadata, per-sample metrics,
queries and, with `--save-scores`, actual coordinates, masks, raw scores,
logits and probabilities for reproducible figures.

Comparisons must disclose query policy, coordinate policy, number of seeds
and checkpoint selection for each method. This runner does not retrain
baselines or supply missing baseline weights/results.
