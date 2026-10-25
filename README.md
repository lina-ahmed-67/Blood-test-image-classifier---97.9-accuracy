# Blood Cell Image Classifier (from-scratch CNN)

A convolutional neural network trained **from scratch** (no pretrained weights) to classify 28x28 RGB blood cell images into 8 classes (labels 0-7). Built in PyTorch for a Kaggle competition scored on categorical accuracy.

> **Note: most of the models in this project are experimental.** The notebook is a record of iterative experiments (different tuning objectives, epoch counts, training data splits, and inference tricks). Several of the saved checkpoints and submission files exist to compare ideas, not as polished final models. See [Experiments](#experiments) for what each one is.

## Repository layout

| Path | Description |
|---|---|
| `bloodcell_classifier.ipynb` | Full pipeline: data prep, model, Optuna tuning, training, prediction |
| `train/`, `test/` | Image folders (`<id>.png`) |
| `train.csv` | Training labels (image id column + `label` column) |
| `*.pt` | Saved model weights (see Experiments) |
| `*_submission.csv` | Prediction files generated from each experiment |

## Data

- 28x28 RGB PNG images, 8 classes with mild imbalance (largest class about 2.7x the smallest).
- Labels come from `train.csv`, so a custom `CellDataset` is used instead of `ImageFolder`.
- 85/15 stratified train/validation split (`random_state=42`).
- Per-channel mean/std are computed from the training split only and reused for validation and test.
- Training augmentation: random horizontal/vertical flips, random 90-degree rotations, and a random crop with reflect padding. Evaluation uses only `ToTensor` + `Normalize`.

## Model

A VGG-style CNN with three stages. Each stage is `Conv3x3 -> BatchNorm -> Activation` twice, followed by `MaxPool(2)`:

```
Input 3x28x28
Stage 1: width channels      -> 14x14
Stage 2: 2 x width channels  -> 7x7
Stage 3: 4 x width channels  -> 3x3
AdaptiveAvgPool -> Dropout -> Linear(4 x width, 8)
```

- Convs use `bias=False` (BatchNorm supplies the shift) and Kaiming initialization.
- The final layer outputs raw logits; softmax is applied only at prediction time.
- Best configuration found: `width=48`, LeakyReLU(0.1), dropout about 0.33.

## Training setup

- Optimizer: AdamW, cosine annealing learning-rate schedule (`T_max` = number of epochs), batch size 32, plain `CrossEntropyLoss`.
- Hyperparameters were tuned with Optuna (40 trials, 20 epochs each, median pruning) over learning rate, weight decay, dropout, batch size, activation and width.

## Experiments

All models share the same architecture; they differ in tuning objective, training length, data, and inference.

| Experiment | Weights file | Submission file | Notes |
|---|---|---|---|
| F1-tuned params, 50 epochs | `f1_model.pt` | `f1_submission.csv` | Optuna objective was validation macro-F1. Checkpoint chosen by lowest validation loss. |
| Accuracy-tuned params, 50 epochs | `accuracy_model.pt` | `accuracy_submission.csv` | Optuna objective was validation accuracy (the competition metric). Best-val-loss checkpoint. |
| Accuracy-tuned + TTA | `accuracy_model.pt` | `tta_accuracy_submission.csv` | Test-time augmentation: average softmax over 8 flip/rotation views. Gains were within noise. |
| Accuracy-tuned, 100 epochs | `accuracy_model_100epochs.pt` | `accuracy_100epochs_submission.csv` | Longer cosine schedule. Best leaderboard result so far (about 97.83%). |
| 100 epochs, train + val data | `accuracy_model_full_dataset.pt` | `accuracy_fulldataset_submission.csv` | Trained on all labeled data, last-epoch weights, no validation available. Experimental. |

### Results at a glance

- Single-model validation accuracy is about 97.8-98.2% depending on the run.
- Leaderboard: about 97.71% (50 epochs) improving to about 97.83% (100 epochs).
- Differences between runs are roughly 0.1 points, which is close to seed-to-seed noise, so individual comparisons should be read cautiously.

## Caveats

- **Experimental status:** these are exploratory runs, not a validated final pipeline. Hyperparameters were tuned on short 20-epoch runs, and the train+val model cannot be validated.
- Validation-based checkpoint selection and Optuna selection both slightly inflate validation scores.
- The `.pt` file for the 50-epoch accuracy run was overwritten when the 100-epoch run reused the same naming pattern earlier in the notebook history; check file timestamps if unsure which model a file holds.
- Test predictions rely on the sorted order of files in `test/`, so do not shuffle the test loader.

## Running it

1. Install dependencies: `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `pillow`, `optuna`.
2. Place `train/`, `test/` and `train.csv` next to the notebook.
3. Run the notebook cells in order. On Windows, `num_workers=0` is used in all DataLoaders to avoid multiprocessing issues.
4. Training 50 epochs at batch size 32 takes roughly 25-30 minutes per model on the hardware used; 100 epochs takes about an hour.

## Ideas not yet tried

- Ensembling several seeds (check competition rules first).
- Label smoothing, EMA of weights, Mixup/CutMix.
- Resizing inputs to 56x56, an extra stage, or hand-written residual connections.
