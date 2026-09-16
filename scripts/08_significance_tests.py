"""Diebold-Mariano tests between every pair of trained models on the test week.

Uses the predictions saved by scripts/06_evaluate_models.py (networks: seed 42), so no model is
retrained. Writes significance.json and significance.md to results/4_model_evaluation/.

Usage (from the repository root, after `python scripts/06_evaluate_models.py`):
    python scripts/08_significance_tests.py
"""
import itertools
import json

import pandas as pd

from milan_forecasting import config
from milan_forecasting.forecasting.diagnostics import load_predictions
from milan_forecasting.forecasting.figures import MODEL_LABELS, TRAINED_MODELS
from milan_forecasting.forecasting.significance import LOSSES, diebold_mariano


def stars(p_value: float) -> str:
    return "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""


def main() -> None:
    runs = pd.read_csv(config.EVALUATION_DIR / "metrics_all_runs.csv")
    squares = list(dict.fromkeys(runs["square"]))

    rows = []
    for square in squares:
        frame = load_predictions(config.FORECASTS_DIR / f"predictions_{square}.csv")
        for model_a, model_b in itertools.combinations(TRAINED_MODELS, 2):
            for loss in LOSSES:
                result = diebold_mariano(frame["actual"], frame[model_a], frame[model_b], loss=loss)
                rows.append({"square": square, "model_a": model_a, "model_b": model_b, **result})
    table = pd.DataFrame(rows)

    lines = [
        "Diebold-Mariano tests on the test week (1,008 one-step forecasts per square; networks: seed 42).",
        "Negative statistic: model A has the lower loss. Newey-West variance, HLN-adjusted, t(n-1) p-values.",
        "* p < 0.05, ** p < 0.01, *** p < 0.001.",
        "",
        "| Square | A vs B | DM, absolute loss | p | DM, squared loss | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (square, model_a, model_b), group in table.groupby(["square", "model_a", "model_b"], sort=False):
        by_loss = group.set_index("loss")
        cells = []
        for loss in LOSSES:
            r = by_loss.loc[loss]
            cells += [f"{r.statistic:.2f}{stars(r.p_value)}", f"{r.p_value:.3g}"]
        lines.append(f"| {square} | {MODEL_LABELS[model_a]} vs {MODEL_LABELS[model_b]} | " + " | ".join(cells) + " |")

    (config.EVALUATION_DIR / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (config.EVALUATION_DIR / "significance.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
