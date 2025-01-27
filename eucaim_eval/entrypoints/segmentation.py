import os
import re
import json
from pathlib import Path
from eucaim_eval import SegmentationMetrics

file_formats = [".nii.gz", ".nii", ".nrrd", ".nhdr"]


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
        "--verbose",
        default=False,
        action="store_true",
        help="Sets verbosity (a progress bar).",
    )

    args = parser.parse_args()

    params = eval(args.params) if args.params is not None else {}
    print(params)

    file_pattern = f"(?<={os.sep})[a-zA-Z0-9_\-\.]*(?={'|'.join(file_formats)})"

    gt_paths = {
        re.search(file_pattern, str(p)).group(): str(p)
        for p in Path(args.gt).rglob("*")
    }
    pred_paths = {
        re.search(file_pattern, str(p)).group(): str(p)
        for p in Path(args.pred).rglob("*")
    }

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
