import os
import re
import json
from pathlib import Path
from typing import Iterator
from eucaim_eval import SegmentationMetrics

file_formats = [".nii.gz", ".nii", ".nrrd", ".nhdr"]
file_pattern = f"(?<={os.sep})[a-zA-Z0-9_\-\.]*(?={'|'.join(file_formats)})"


def fetch_paths(path: str) -> Iterator[tuple[str, str]]:
    for p in Path(path).rglob("*"):
        file_search = re.search(file_pattern, str(p))
        if file_search is not None:
            yield str(p), file_search.group()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Path to folder with predictions.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Path to folder with ground truth.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        required=True,
        help="Number of classes.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        choices=SegmentationMetrics().AVAILABLE_METRICS,
        nargs="+",
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence interval.",
    )
    parser.add_argument(
        "--input_is_one_hot",
        default=False,
        action="store_true",
        help="Whether the input is one hot.",
    )
    parser.add_argument(
        "--params",
        default=None,
        type=str,
        help="Parameters for the metrics. Should be specified as a dictionary.",
    )
    parser.add_argument(
        "--per_lesion",
        default=False,
        action="store_true",
        help="Matches predictions to the most likely ground truth.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of workers.",
    )
    parser.add_argument(
        "--cache",
        type=int,
        default=0,
        help="Cache size (recommended to set to small number here).",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Sets verbosity (a progress bar).",
    )

    args = parser.parse_args()

    params = eval(args.params) if args.params is not None else {}

    gt_paths = {k: p for p, k in fetch_paths(args.gt)}
    pred_paths = {k: p for p, k in fetch_paths(args.pred)}

    preds, gts = [], []
    for key in gt_paths.keys():
        if key in pred_paths:
            preds.append(pred_paths[key])
            gts.append(gt_paths[key])

    seg_metrics = SegmentationMetrics(
        n_classes=args.n_classes,
        n_workers=args.n_workers,
        verbose=args.verbose,
        match_regions=args.per_lesion,
        input_is_one_hot=args.input_is_one_hot,
        params=params,
        ci=args.ci,
        metrics=args.metrics,
        cache_size=args.cache,
    )

    metrics = seg_metrics.calculate_metrics(preds=preds, gts=gts)

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
    else:
        from pprint import pprint

        pprint(metrics)


if __name__ == "__main__":
    main()
