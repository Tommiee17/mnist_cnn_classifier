# MNIST CNN Classifier

Simple PyTorch project for classifying MNIST digits (`0-9`) with CNN.

## What This Project Does

- Downloads and loads MNIST 
- Trains a CNN classifier
- Evaluates on validation and test data
- Saves:
  - model weights to `models/mnist_cnn.pt`
  - training history to `output/history/train_history.json`
  - test outputs to `output/history/test_data.json`
- Plots training curves and confusion matrix

## Project Structure

```text
mnist/
├── load_data.py         # MNIST loaders
├── model.py             # CNN_Classifier
├── train.py             # train loop (one epoch)
├── eval.py              # evaluation loop
├── trainer.py           # full train/load pipeline
└── visualization.py     # curves + confusion matrix

run.py                   # entry point
models/                  # saved checkpoints
output/history/          # saved metrics/predictions
data/                    # MNIST data
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From repo root:

```bash
python run.py
```

`run.py` calls `trainer(load_model=...)`:
- `load_model=True`: loads existing checkpoint + history + test data
- `load_model=False`: trains a new model and saves outputs

For first run, set `load_model=False` in `run.py`.

## Outputs

- `models/mnist_cnn.pt`: model weights (`state_dict`)
- `output/history/train_history.json`: per-epoch metrics
- `output/history/test_data.json`: test loss, test accuracy, labels, predictions
- Plot image from `plot_curves(...)` (saved via `mnist/visualization.py`)

