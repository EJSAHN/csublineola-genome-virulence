from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--x-key", required=True)
    parser.add_argument("--y-key", required=True)
    parser.add_argument("--reps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arrays = np.load(args.input)
    x = arrays[args.x_key]
    y = arrays[args.y_key]
    result = stats.multiscale_graphcorr(
        x,
        y,
        compute_distance=None,
        reps=args.reps,
        workers=1,
        random_state=args.seed,
    )
    payload = {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "optimal_scale": [int(v) for v in result.mgc_dict["opt_scale"]],
        "reps": int(args.reps),
        "seed": int(args.seed),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
