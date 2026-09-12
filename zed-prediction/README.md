# zed-prediction

Working folder for the ICRA prediction paper. **Runnable code stays at the root** (same as before on the Jetson). Extra folders are only for paper assets.

## Layout

| Path | Contents |
|------|----------|
| `zed2.py`, `ekf_zed.py`, `plot_eval.py`, `test_config.py`, … | Run these here (flat, like before) |
| `results/` | CSV logs + dashboard PNGs (`paper_1s/` = clean 1 s set) |
| `related_papers/` | PDFs of related work |
| `draft1/` | Paper draft 1 |
| `docs/` | Notes (`plan.txt`, `motion_models.md`) |

Nav2 / SLAM YAML files for the robot stay in the **PaperIcra repo root** (unchanged), not inside this folder.

## Run (Mac or Jetson — same commands)

```bash
cd ~/PaperIcra/zed-prediction   # or your clone path after git pull
# edit test_config.py as usual
python3 zed2.py
python3 plot_eval.py
```

Optional live RViz helpers:

```bash
source /opt/ros/humble/setup.bash
python3 danger_zone_viz.py
# or with prediction: python3 zed2.py  (uses danger_zone_ros when wired)
```

SVO paths in `test_config.py` stay relative, e.g. `recordings/D-walk.svo2` next to this folder.

## Sync

```bash
# on this machine
git add -A && git commit -m "..." && git push

# on main system (Jetson)
cd ~/PaperIcra && git pull
cd zed-prediction && python3 zed2.py
```
