#!/usr/bin/env bash
# Run the COCO benchmark N times (default 3) and write per-row medians to
# benches/results/coco300.json (raw runs: benches/results/coco300.run*.raw.json).
# The machine is shared/noisy, so medians are more honest than a single run.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$PROJECT_DIR"
N=${RUNS:-3}
for i in $(seq 1 "$N"); do
  python benches/bench_vs_albumentations.py --data "$AUGRS_DATA" --threads "${THREADS:-4}" \
    --out "benches/results/coco300.run$i.raw.json" "$@" | grep -v "^ "
done
python - "$N" <<'PY'
import json, statistics, sys
n = int(sys.argv[1])
runs = [json.load(open(f"benches/results/coco300.run{i}.raw.json")) for i in range(1, n + 1)]
out = dict(runs[0])
out["runs"] = n
rows = []
for k, row in enumerate(runs[0]["rows"]):
    vals = [r["rows"][k]["images_per_s"] for r in runs]
    rows.append({**row, "images_per_s": statistics.median(vals), "all_runs": vals})
out["rows"] = rows
json.dump(out, open("benches/results/coco300.json", "w"), indent=2)
for r in rows:
    print(f"  {r['pipeline']:12s} {r['config']:48s} {r['images_per_s']:8.1f}  {[round(v) for v in r['all_runs']]}")
PY
