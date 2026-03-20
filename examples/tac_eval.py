from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from truthanchor import (
    compute_uncertainty_scores,
    evaluate_saved_mappers,
    generate_responses,
    prepare_custom_uncertainty_scores,
    train_mappers,
)
from truthanchor.utils.io import load_jsonl
from truthanchor.utils.io import load_npz_dict
from truthanchor.utils.methods import METHOD_INFO, METHODS_PLOT
from truthanchor.utils.paths import artifact_dir
from truthanchor.utils.visualization import (
    plot_auc_comparison,
    plot_auc_comparison_3way,
    plot_calibration_diagram,
    plot_ece_comparison,
    plot_ece_comparison_3way,
)


DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]

DEFAULT_DATASETS = [
    "trivia",
    "sciq",
    "popqa",
]


def resolve_method_info(method: str, out_dir: Path):
    if method in METHOD_INFO:
        return METHOD_INFO[method]
    metadata_path = out_dir / "custom_methods.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method_spec = metadata.get("methods", {}).get(method)
        if method_spec is not None:
            return method_spec["saved_key"], bool(method_spec["higher_worse"])
    raise KeyError(f"Unknown uncertainty method '{method}'.")


def run_pipeline(
    model_name: str,
    dataset_name: str,
    ue_methods: list[str],
    output_root: str,
    max_new_tokens: int,
    num_samples: int,
    batch_size: int,
    data_portion: float,
    rank_weight: float,
    use_cue: bool,
    plot_comparison: bool,
    plot_calibration: bool,
    custom_scores_csv: str | None,
    custom_method_name: str,
    higher_worse: bool | None,
):
    out_dir = artifact_dir(output_root, dataset_name, model_name)

    print("=== Running TAC Eval ===")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"UE methods: {' '.join(ue_methods)}")
    print(f"Output directory: {out_dir}")

    if custom_scores_csv is not None:
        if higher_worse is None:
            raise ValueError("--higher_worse must be provided when using --custom_scores_csv.")
        if use_cue:
            raise ValueError("CUE training requires prompt texts and is not supported for custom score-label CSV input.")
        npz_path, metadata_path = prepare_custom_uncertainty_scores(
            dataset_name=dataset_name,
            higher_worse=higher_worse,
            csv_path=custom_scores_csv,
            output_root=output_root,
            model_name=model_name,
            method_name=custom_method_name,
        )
        print("Saved custom uncertainty scores to:")
        print(f"  {npz_path}")
        print("Saved custom method metadata to:")
        print(f"  {metadata_path}")
        uncertainty_data = load_npz_dict(npz_path)
        texts = None
        if custom_method_name not in ue_methods:
            ue_methods = [custom_method_name]
    else:
        # generate_responses(
        #     model_name=model_name,
        #     dataset_name=dataset_name,
        #     max_new_tokens=max_new_tokens,
        #     data_portion=data_portion,
        #     num_samples=num_samples,
        #     save=True,
        #     output_root=output_root,
        #     batch_size=batch_size,
        # )
        # print("Saved generation artifacts to:")
        # print(f"  {out_dir / f'responses_{dataset_name}.jsonl'}")
        # print(f"  {out_dir / 'generation_results.npz'}")

        compute_uncertainty_scores(
            model_name=model_name,
            dataset_name=dataset_name,
            output_root=output_root,
        )
        print("Saved uncertainty scores to:")
        print(f"  {out_dir / 'uncertainty_scores.npz'}")
        uncertainty_data = load_npz_dict(out_dir / "uncertainty_scores.npz")
        texts = np.asarray([row["prompt"] for row in load_jsonl(out_dir / f"responses_{dataset_name}.jsonl")])

    results_dir = out_dir / "mapper_eval"
    scores_dir = results_dir / "scores"
    results_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "val_auroc_orig",
                "val_ece_orig",
                "val_prr_orig",
                "val_auroc_mapped",
                "val_ece_mapped",
                "val_prr_mapped",
                "test_auroc_orig",
                "test_ece_orig",
                "test_prr_orig",
                "test_auroc_mapped",
                "test_ece_mapped",
                "test_prr_mapped",
                "val_auroc_cue",
                "val_ece_cue",
                "val_prr_cue",
                "test_auroc_cue",
                "test_ece_cue",
                "test_prr_cue",
            ],
        )
        writer.writeheader()
        orig_eces = []
        mapped_eces = []
        cue_eces = []
        orig_aucs = []
        mapped_aucs = []
        cue_aucs = []
        plotted_methods = []

        y_all = np.asarray(uncertainty_data["labels"])
        indices = np.arange(len(y_all))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y_all,
        )

        for method in ue_methods:
            key, method_higher_worse = resolve_method_info(method, out_dir)
            X = np.asarray(uncertainty_data[key]).reshape(-1, 1)
            if method_higher_worse:
                X = -X
            y = y_all
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            texts_train = texts[train_idx] if texts is not None else None
            texts_test = texts[test_idx] if texts is not None else None

            mapper, corrector, val_report = train_mappers(
                X_train,
                y_train,
                rank_weight=rank_weight,
                use_cue=use_cue,
                texts_train=texts_train,
            )
            test_report = evaluate_saved_mappers(
                mapper,
                X_test,
                y_test,
                corrector=corrector if use_cue else None,
                texts_test=texts_test if use_cue else None,
                cue_w=val_report.get("cue_w"),
                cue_scaler=val_report.get("cue_scaler"),
            )

            print(
                f"{method}: val mapped AUROC = {val_report['auroc_mapped']:.4f}, ECE = {val_report['ece_mapped']:.4f}, PRR = {val_report['prr_mapped']:.4f}; "
                f"val orig AUROC = {val_report['auroc_orig']:.4f}, ECE = {val_report['ece_orig']:.4f}, PRR = {val_report['prr_orig']:.4f}"
            )
            if use_cue:
                print(
                    f"{method}: val CUE AUROC = {val_report['auroc_cue']:.4f}, ECE = {val_report['ece_cue']:.4f}, PRR = {val_report['prr_cue']:.4f}"
                )
            print(
                f"{method}: test mapped AUROC = {test_report['auroc_mapped']:.4f}, ECE = {test_report['ece_mapped']:.4f}, PRR = {test_report['prr_mapped']:.4f}; "
                f"test orig AUROC = {test_report['auroc_orig']:.4f}, ECE = {test_report['ece_orig']:.4f}, PRR = {test_report['prr_orig']:.4f}"
            )
            if use_cue:
                print(
                    f"{method}: test CUE AUROC = {test_report['auroc_cue']:.4f}, ECE = {test_report['ece_cue']:.4f}, PRR = {test_report['prr_cue']:.4f}"
                )

            if method in METHODS_PLOT:
                plotted_methods.append(METHODS_PLOT[method])
                orig_eces.append(test_report["ece_orig"])
                mapped_eces.append(test_report["ece_mapped"])
                orig_aucs.append(test_report["auroc_orig"])
                mapped_aucs.append(test_report["auroc_mapped"])
                if use_cue:
                    cue_eces.append(test_report["ece_cue"])
                    cue_aucs.append(test_report["auroc_cue"])

            if plot_calibration and method in METHODS_PLOT:
                plot_calibration_diagram(
                    test_report["raw_scores"],
                    test_report["labels"],
                    test_report["auroc_orig"],
                    test_report["ece_orig"],
                    METHODS_PLOT[method],
                    results_dir,
                    anchored=0,
                )
                plot_calibration_diagram(
                    test_report["mapped_scores"],
                    test_report["labels"],
                    test_report["auroc_mapped"],
                    test_report["ece_mapped"],
                    METHODS_PLOT[method],
                    results_dir,
                    anchored=1,
                )
                if use_cue:
                    plot_calibration_diagram(
                        test_report["cue_scores"],
                        test_report["labels"],
                        test_report["auroc_cue"],
                        test_report["ece_cue"],
                        METHODS_PLOT[method] + "_CUE",
                        results_dir,
                        anchored=2,
                    )

            np.savez_compressed(
                scores_dir / f"{method}_validation_scores.npz",
                labels=val_report["labels"],
                raw_scores=val_report["raw_scores"],
                mapped_scores=val_report["mapped_scores"],
                cue_scores=val_report.get("cue_scores"),
            )
            np.savez_compressed(
                scores_dir / f"{method}_test_scores.npz",
                labels=test_report["labels"],
                raw_scores=test_report["raw_scores"],
                mapped_scores=test_report["mapped_scores"],
                cue_scores=test_report.get("cue_scores"),
            )

            writer.writerow(
                {
                    "method": method,
                    "val_auroc_orig": val_report["auroc_orig"],
                    "val_ece_orig": val_report["ece_orig"],
                    "val_prr_orig": val_report["prr_orig"],
                    "val_auroc_mapped": val_report["auroc_mapped"],
                    "val_ece_mapped": val_report["ece_mapped"],
                    "val_prr_mapped": val_report["prr_mapped"],
                    "test_auroc_orig": test_report["auroc_orig"],
                    "test_ece_orig": test_report["ece_orig"],
                    "test_prr_orig": test_report["prr_orig"],
                    "test_auroc_mapped": test_report["auroc_mapped"],
                    "test_ece_mapped": test_report["ece_mapped"],
                    "test_prr_mapped": test_report["prr_mapped"],
                    "val_auroc_cue": val_report.get("auroc_cue"),
                    "val_ece_cue": val_report.get("ece_cue"),
                    "val_prr_cue": val_report.get("prr_cue"),
                    "test_auroc_cue": test_report.get("auroc_cue"),
                    "test_ece_cue": test_report.get("ece_cue"),
                    "test_prr_cue": test_report.get("prr_cue"),
                }
            )

    if plot_comparison and plotted_methods:
        if use_cue:
            plot_ece_comparison_3way(dataset_name, plotted_methods, orig_eces, mapped_eces, cue_eces, results_dir)
            plot_auc_comparison_3way(dataset_name, plotted_methods, orig_aucs, mapped_aucs, cue_aucs, results_dir)
        else:
            plot_ece_comparison(dataset_name, plotted_methods, orig_eces, mapped_eces, results_dir)
            plot_auc_comparison(dataset_name, plotted_methods, orig_aucs, mapped_aucs, results_dir)

    print("Saved mapper evaluation results to:")
    print(f"  {results_path}")
    print("Saved validation and test score files to:")
    print(f"  {scores_dir}")
    if plot_comparison:
        print("Saved comparison plots to:")
        if use_cue:
            print(f"  {results_dir / f'{dataset_name}_ece_comparison_3way.png'}")
            print(f"  {results_dir / f'{dataset_name}_auc_comparison_3way.png'}")
        else:
            print(f"  {results_dir / f'{dataset_name}_ece_comparison.png'}")
            print(f"  {results_dir / f'{dataset_name}_auc_comparison.png'}")
    if plot_calibration:
        print("Saved per-method calibration plots to:")
        print(f"  {results_dir / 'calibration_*.png'}")
    print()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--ue_methods", nargs="+", default=list(METHOD_INFO))
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_portion", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--use_cue", action="store_true")
    parser.add_argument("--plot_comparison", action="store_true")
    parser.add_argument("--plot_calibration", action="store_true")
    parser.add_argument("--custom_scores_csv", type=str, default=None)
    parser.add_argument("--custom_method_name", type=str, default="custom_score")
    parser.add_argument("--higher_worse", type=lambda x: x.lower() in {"1", "true", "yes"}, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for model_name in args.models:
        for dataset_name in args.datasets:
            run_pipeline(
                model_name=model_name,
                dataset_name=dataset_name,
                ue_methods=args.ue_methods,
                output_root=args.output_root,
                max_new_tokens=args.max_new_tokens,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                data_portion=args.data_portion,
                rank_weight=args.rank_weight,
                use_cue=args.use_cue,
                plot_comparison=args.plot_comparison,
                plot_calibration=args.plot_calibration,
                custom_scores_csv=args.custom_scores_csv,
                custom_method_name=args.custom_method_name,
                higher_worse=args.higher_worse,
            )


if __name__ == "__main__":
    main()
