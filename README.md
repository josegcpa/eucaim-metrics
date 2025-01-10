# EUCAIM metrics

Here, we implement the set of metrics which will be used to evaluate the performance of models within the EUCAIM project.

So far we have implemented metric sets for the following tasks:
* **Segmentation**: Dice score, intersection over union, Hausdorff distance, and normalised surface distance.
* **Classification**: Accuracy, precision, recall, F1-score, ROC-AUC and average precisionn.

The selection of metrics and small parts of the implementation were based on the [Metrics Reloaded](https://metrics-reloaded.dkfz.de/) project.

## TODO

* Detection metrics
* Detection metrics for segmentation models
* Calibration metrics
* Regression metrics
* Adapt segmentation metrics to multiprocessing
* Complete documentation