# EUCAIM eval

Here, we implement the set of metrics which will be used to evaluate the performance of models within the EUCAIM project.

So far we have implemented metric sets for the following tasks:
* **Segmentation**: Dice score, intersection over union, Hausdorff distance, and normalised surface distance.
* **Classification**: Accuracy, precision, recall, F1-score, ROC-AUC and average precisionn.

The selection of metrics and small parts of the implementation were based on the [Metrics Reloaded](https://metrics-reloaded.dkfz.de/) project.

## Usage

We use [`uv`](https://github.com/astral-sh/uv) as a dependency manager and for running scripts. So using this package can be done through a `pip` installation or through `uv` commands

### Installation as a pip package

To install this package as a pip package, simply run `uv pip install -r pyproject.toml`.

### Usage through `uv` commands

`uv` is capable of doing all of the dependency heavy lifting upon running (similarly to `cargo` for Rust). So you can simply download this and, to execute the segmentation evaluation, simply run `uv run eval_segmentation`. For now this is the only entrypoint (details below).

### CLI entrypoints

#### `eval_segmentation`

Evaluates segmentation metrics for a given set of predictions and ground truth. An example of this is:

```bash
uv run \
    eval_segmentation \
    --pred test_data/predicted/ \
    --gt test_data/groundtruth/ \
    --output test.json \
    --n_classes 2 \
    --verbose \
    --params '{"normalised_surface_distance": {"max_distance": 100.0}}'
```

And a commented version:

```bash
uv run \
    eval_segmentation \ # the name of the entrypoint
    --pred test_data/predicted/ \ # the path to the prediction folder
    --gt test_data/groundtruth/ \ # the path to the ground truth folder
    --output test.json \ # the path to the output file
    --n_classes 2 \ # the number of classes
    --verbose \ # uses a progress bar to track metrics
    --params '{"normalised_surface_distance": {"max_distance": 100.0}}' # the parameters for the metrics
```

When specifying either `--pred` or `--gt`, this assumes that predictions and ground truths are identically named (excluding the file format). Picks up only specific file formats.

## TODO

* Detection metrics
* Detection metrics for segmentation models
* Calibration metrics
* Regression metrics
[x] Adapt segmentation metrics to multiprocessing
[x] Adapt segmentation metrics to multiple objects of the same class
* Complete documentation