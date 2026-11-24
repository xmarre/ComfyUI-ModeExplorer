#!/usr/bin/env python3
"""Compare a teacher and a student activation probe report (student / teacher ratios per block).

    python -m tools.rediverse_probe_report teacher.json student.json [--json out.json]

Reports come from the ModeExplorer Activation Probe node
(output/modeexplorer_rediverse/probes). The collapsed-block rule follows ReDiverse
App. A: the block where the concentration ratio steps up the most.
"""

import argparse
import json
from pathlib import Path

from mode_explorer.rediverse_runtime import compare_probes


def table(result):
    lines = ["block     PR ratio   TV ratio   Conc ratio   erank ratio"]
    fmt = lambda v: f"{v:10.3f}" if isinstance(v, (int, float)) else f"{'n/a':>10}"
    for name, row in result["blocks"].items():
        lines.append(f"{name:8}{fmt(row['pr_ratio'])} {fmt(row['tv_ratio'])} {fmt(row['conc_ratio'])}   {fmt(row['entropy_rank_ratio'])}")
    lines.append(f"largest concentration step: {result['largest_concentration_step']}")
    lines.append(f"matched inputs: {result['matched_inputs']}")
    lines.append(result["caveat"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("student", type=Path)
    parser.add_argument("--json", type=Path, help="Also write the comparison as JSON.")
    args = parser.parse_args()
    result = compare_probes(json.loads(args.teacher.read_text()), json.loads(args.student.read_text()))
    print(table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
