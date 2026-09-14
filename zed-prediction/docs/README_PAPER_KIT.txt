PAPER OFFLINE KIT — write on Mac without Jetson
================================================
Folder: src/zed-detection/zed-prediction/docs/

Copy this whole `docs/` folder (or the git repo) to your Mac.
You do NOT need ROS, ZED, or the robot to finish a Block-1 paper draft.

------------------------------------------------
1. WHAT YOU HAVE (ready to cite / insert)
------------------------------------------------

### Claim (safe for paper)
On two ZED SVO walks, a hip-steering EKF 1 s open-loop forecast has lower
mean ADE than CTRV (same filter, same measurements; only motion model changes).

Do NOT claim: Nav2 is safer with prediction (Block 2 not proven).

### Locked 1 s numbers (match draft1/paper/root.tex Table I)
From paper draft / paper_1s logs:

  D-walk   CTRV ADE=0.203  Hip ADE=0.192   N=537
  Straight CTRV ADE=0.206  Hip ADE=0.198   N=232
  FDE (last scored frame):
  D-walk   CTRV=0.502  Hip=0.528
  Straight CTRV=0.111  Hip=0.093

### Extra baselines (from block1_ade_table — useful if you expand Table I)
  D-walk   Static≈0.634  CV≈0.256  CTRV≈0.203  Hip≈0.192
  Straight Static≈0.800  CV≈0.254  CTRV≈0.206  Hip≈0.198
  (n slightly different clip window; if you use these, recompute from CSVs
   in data/ and state N clearly.)

### Method one-liners
  Sensor: ZED BODY_18, hip KP 8 & 11
  State:  [px, py, v, theta, omega]
  Meas:   [px, py, theta_hip]
  CTRV:   turn with omega
  Hip:    theta += b*sin(theta_hip - theta)*dt, then step with v; b=3.0
  Forecast: T=1.0 s, 20 open-loop steps after update
  Coords: px = cam Z (forward), py = cam X (lateral)

### Noise used in paper draft
  P0 = diag(1,1,100,10,5)
  Q  = diag(0.01,0.01,0.1,0.05,0.01)
  R  = diag(0.1,0.1,1.0)

------------------------------------------------
2. FILES IN THIS KIT
------------------------------------------------

docs/
  README_PAPER_KIT.txt          ← this file
  CLAIM_AND_LIMITS.txt          ← what you may / may not claim
  LATEX_FIGURE_SNIPPETS.tex     ← copy-paste into root.tex
  METHOD_CHEATSHEET.txt         ← equations / symbols
  figures/
    fig_ade_bars.png            ← main ADE bar chart
    fig_dwalk_overlay.png       ← D-walk overlay (use as Fig.1)
    fig_error_over_time.png
    fig_trajectories.png
    ekf_eval_dashboard_hip_Dwalk_1s.png
    ekf_eval_dashboard_ctrv_Dwalk_1s.png
    ekf_eval_dashboard_hip_straight_1s.png
    ekf_eval_dashboard_ctrv_straight_1s.png
  data/
    block1_ade_table.txt / .csv
    ekf_prediction_log_hip_Dwalk_1s.csv
    ekf_prediction_log_ctrv_Dwalk_1s.csv
    ekf_prediction_log_hip_straight_1s.csv
    ekf_prediction_log_ctrv_straight_1s.csv
  paper_snippets/
    root.tex                    ← current draft
    refs.bib
    IEEEabrv.bib

------------------------------------------------
3. TODO ON MAC (paper writing checklist)
------------------------------------------------

[ ] Insert fig_dwalk_overlay.png (replace placeholder in root.tex)
[ ] Optionally add fig_ade_bars.png as second figure
[ ] Soften abstract: Nav2 = future work / prototype, not a result
[ ] Keep conclusion: no closed-loop Nav2 comparison
[ ] Optional: add Static+CV columns using block1_ade_table (recheck N)
[ ] Compile with IEEEtran / ieeeconf on Overleaf
[ ] Do NOT wait for Block 2 A/B to submit a prediction-only paper

------------------------------------------------
4. BLOCK 2 STATUS (for discussion / future work only)
------------------------------------------------

Live Nav2 person cloud + path guard exist on the Jetson, but logger A/B
did NOT show prediction clearly safer than "now". Treat as unfinished
system work, not a paper result.

------------------------------------------------
5. HOW TO GET THIS ONTO THE MAC
------------------------------------------------

From Jetson (while networked):
  scp -r user@JETSON:~/ros2_ws/src/zed-detection/zed-prediction/docs ~/Desktop/zed-paper-docs

Or copy the whole zed-prediction folder / git push then pull on Mac.

On Mac Overleaf: upload figures/*.png + paper_snippets/root.tex + refs.bib
