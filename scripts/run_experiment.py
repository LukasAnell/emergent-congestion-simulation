"""Run the congestion simulation density sweep."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.experiment import run_density_sweep, run_turn_sweep  # noqa: E402
from src.plotting import plot_summary, plot_turn_sweep_heatmap  # noqa: E402
from src.utils import ensure_dir, parse_density_list  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run congestion density sweep.")
    parser.add_argument("--config", type=str, default="configs/base.json", help="Path to config JSON")
    parser.add_argument("--N", type=int, help="Grid size N")
    parser.add_argument("--densities", type=str, help="Comma-separated densities list")
    parser.add_argument("--burn-in-steps", type=int, help="Burn-in steps")
    parser.add_argument("--measurement-steps", type=int, help="Measurement steps")
    parser.add_argument("--replications", type=int, help="Number of replications")
    parser.add_argument("--seed-base", type=int, help="Seed base")
    parser.add_argument("--p-turn", type=float, help="Turning probability (0 to 1)")
    parser.add_argument("--p-turn-values", type=str, help="Comma-separated p_turn sweep values")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    if args.N is not None:
        config.N = args.N
    if args.densities:
        config.densities = parse_density_list(args.densities)
    if args.burn_in_steps is not None:
        config.burn_in_steps = args.burn_in_steps
    if args.measurement_steps is not None:
        config.measurement_steps = args.measurement_steps
    if args.replications is not None:
        config.replications = args.replications
    if args.seed_base is not None:
        config.seed_base = args.seed_base
    if args.p_turn is not None:
        config.p_turn = args.p_turn
    if args.p_turn_values:
        config.p_turn_values = parse_density_list(args.p_turn_values)
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    output_dir = ensure_dir(config.output_dir)

    summary_df = run_density_sweep(config)

    turn_sweep_df = run_turn_sweep(config) if config.p_turn_values else None

    plot_dir = None
    if not args.no_plots:
        plot_dir = ensure_dir(Path(output_dir) / "plots")
        plot_summary(Path(output_dir) / "summary.csv", plot_dir)
        if turn_sweep_df is not None and not turn_sweep_df.empty:
            plot_turn_sweep_heatmap(
                Path(output_dir) / "turn_sweep_speed_matrix.csv",
                Path(plot_dir) / "speed_heatmap_density_vs_pturn.png",
            )

    print(f"Wrote summary to {Path(output_dir) / 'summary.csv'}")
    if turn_sweep_df is not None:
        print(f"Wrote p_turn sweep to {Path(output_dir) / 'turn_sweep.csv'}")


if __name__ == "__main__":
    main()
