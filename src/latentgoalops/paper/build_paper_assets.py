"""Build the full frozen paper asset bundle in one command."""

from __future__ import annotations

import argparse
import json

from latentgoalops.analysis.make_case_studies import generate_case_studies
from latentgoalops.analysis.make_paper_figures import generate_figures
from latentgoalops.analysis.make_paper_tables import generate_tables
from latentgoalops.paper.export_latex_tables import export_latex_tables
from latentgoalops.paper.run_paper_suite import generate_paper_suite


def build_paper_assets(config_path: str) -> dict:
    suite = generate_paper_suite(config_path)
    tables = generate_tables(config_path)
    figures = generate_figures(config_path)
    cases = generate_case_studies(config_path)
    latex_tables = export_latex_tables(config_path)
    return {
        "suite": suite,
        "tables": tables,
        "figures": {
            "figures_dir": figures["figures_dir"],
            "figure_count": figures["figure_count"],
        },
        "latex_tables": latex_tables,
        "cases": {
            "case_studies_dir": cases["case_studies_dir"],
            "case_count": cases["case_count"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_protocol.yaml")
    args = parser.parse_args()
    print(json.dumps(build_paper_assets(args.config), indent=2))


if __name__ == "__main__":
    main()
