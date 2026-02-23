#!/bin/bash
# ==============================================================================
# Optimizer Comparison Experiment Sweep
# Per experiment plan: experiments/optimizer_comparison.md
# Hardware: RTX PRO 6000 (96GB GDDR7, ~405 TFLOPS BF16)
# Scale: d12 calibration → d20 formal comparison & ablation (~430M params)
# Strategy: Sequential search — primary axes first, then refine at best config
# ==============================================================================
set -euo pipefail

# ---- Configuration ----
export OMP_NUM_THREADS=1
DEPTH_CAL=12        # d12 for hyperparameter calibration
DEPTH_MAIN=20       # d20 for formal comparison & ablation
SEED=42
EVAL_EVERY_CAL=100  # evaluation frequency (calibration: short runs)
EVAL_EVERY_MAIN=250 # evaluation frequency (main: full runs)
CAL_STEPS=500       # calibration run length
# Pro 6000 (96GB) can handle larger device batch size → fewer grad accum steps → faster
DEVICE_BS_CAL=64    # device batch size for d12 calibration
DEVICE_BS_MAIN=64   # device batch size for d20 main runs
COMMON="--seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --core-metric-every=-1 --sample-every=-1 --save-every=-1"

# ==============================================================================
# Phase 1: d12 Hyperparameter Calibration (~5-6 hrs, ~43 runs)
#
# Strategy: sweep primary axes (LR × WD/PF) first with sensible defaults;
# then refine secondary axes (beta2, warmup) at best primary config.
# ==============================================================================

echo "======================================================================"
echo "Phase 1: d12 Hyperparameter Calibration"
echo "======================================================================"

# --------------------------------------------------------------------------
# 1a. MuonAdamW: full grid (12 runs: 4 LR × 3 WD)
#     embedding_lr=0.3 fixed per §4.1
# --------------------------------------------------------------------------
echo "--- 1a. MuonAdamW calibration (12 runs) ---"
for MATRIX_LR in 0.01 0.02 0.03 0.05; do
  for WD in 0.1 0.2 0.3; do
    RUN_NAME="cal_muon_lr${MATRIX_LR}_wd${WD}"
    echo ">> ${RUN_NAME}"
    python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
      --device-batch-size=${DEVICE_BS_CAL} \
      --optimizer-type=muon --matrix-lr=${MATRIX_LR} --weight-decay=${WD} \
      --embedding-lr=0.3 \
      ${COMMON} --run="${RUN_NAME}"
  done
done

# --------------------------------------------------------------------------
# 1b. AdamW: primary grid (15 runs: 5 LR × 3 WD, beta2=0.95 default)
# --------------------------------------------------------------------------
echo "--- 1b. AdamW primary calibration (15 runs) ---"
for LR in 1e-4 2e-4 3e-4 5e-4 8e-4; do
  for WD in 0.01 0.05 0.1; do
    RUN_NAME="cal_adamw_lr${LR}_wd${WD}"
    echo ">> ${RUN_NAME}"
    python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
      --device-batch-size=${DEVICE_BS_CAL} \
      --optimizer-type=adamw --matrix-lr=${LR} --weight-decay=${WD} \
      --adam-beta2=0.95 \
      ${COMMON} --run="${RUN_NAME}"
  done
done

# --------------------------------------------------------------------------
# 1c. AdamW beta2 refinement (2 runs: best LR/WD × beta2 {0.95, 0.99})
#     >>> PAUSE: set BEST_ADAMW_LR and BEST_ADAMW_WD from 1b results <<<
# --------------------------------------------------------------------------
echo "--- 1c. AdamW beta2 refinement (2 runs) ---"
BEST_ADAMW_LR=3e-4   # TODO: set from 1b
BEST_ADAMW_WD=0.1     # TODO: set from 1b
for BETA2 in 0.95 0.99; do
  RUN_NAME="cal_adamw_beta2_${BETA2}"
  echo ">> ${RUN_NAME}"
  python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
    --device-batch-size=${DEVICE_BS_CAL} \
    --optimizer-type=adamw --matrix-lr=${BEST_ADAMW_LR} --weight-decay=${BEST_ADAMW_WD} \
    --adam-beta2=${BETA2} \
    ${COMMON} --run="${RUN_NAME}"
done

# --------------------------------------------------------------------------
# 1d. SOAP: primary grid (8 runs: 4 LR × 2 PF, WD=0.01, warmup=0.05)
# --------------------------------------------------------------------------
echo "--- 1d. SOAP primary calibration (8 runs) ---"
for LR in 1e-3 3e-3 5e-3 1e-2; do
  for PF in 10 50; do
    RUN_NAME="cal_soap_lr${LR}_pf${PF}"
    echo ">> ${RUN_NAME}"
    python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
      --device-batch-size=${DEVICE_BS_CAL} \
      --optimizer-type=soap --soap-lr=${LR} --precondition-frequency=${PF} \
      --weight-decay=0.01 --warmup-ratio=0.05 \
      ${COMMON} --run="${RUN_NAME}"
  done
done

# --------------------------------------------------------------------------
# 1e. SOAP secondary refinement (6 runs)
#     Best LR/PF from 1d → sweep WD, warmup, beta2
#     >>> PAUSE: set BEST_SOAP_LR and BEST_SOAP_PF from 1d results <<<
# --------------------------------------------------------------------------
echo "--- 1e. SOAP secondary refinement (6 runs) ---"
BEST_SOAP_LR=3e-3    # TODO: set from 1d
BEST_SOAP_PF=10      # TODO: set from 1d
# WD refinement (2 runs)
for WD in 0.0 0.01; do
  RUN_NAME="cal_soap_wd${WD}"
  echo ">> ${RUN_NAME}"
  python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
    --device-batch-size=${DEVICE_BS_CAL} \
    --optimizer-type=soap --soap-lr=${BEST_SOAP_LR} --precondition-frequency=${BEST_SOAP_PF} \
    --weight-decay=${WD} --warmup-ratio=0.05 \
    ${COMMON} --run="${RUN_NAME}"
done
# Warmup refinement (2 runs — 0.05 already covered in primary)
for WARMUP in 0.0 0.1; do
  RUN_NAME="cal_soap_warmup${WARMUP}"
  echo ">> ${RUN_NAME}"
  python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
    --device-batch-size=${DEVICE_BS_CAL} \
    --optimizer-type=soap --soap-lr=${BEST_SOAP_LR} --precondition-frequency=${BEST_SOAP_PF} \
    --weight-decay=0.01 --warmup-ratio=${WARMUP} \
    ${COMMON} --run="${RUN_NAME}"
done
# Beta2 refinement (2 runs)
for BETA2 in 0.95 0.99; do
  RUN_NAME="cal_soap_beta2_${BETA2}"
  echo ">> ${RUN_NAME}"
  python -m scripts.base_train --depth=${DEPTH_CAL} --num-iterations=${CAL_STEPS} --eval-every=${EVAL_EVERY_CAL} \
    --device-batch-size=${DEVICE_BS_CAL} \
    --optimizer-type=soap --soap-lr=${BEST_SOAP_LR} --precondition-frequency=${BEST_SOAP_PF} \
    --weight-decay=0.01 --warmup-ratio=0.05 --soap-beta2=${BETA2} \
    ${COMMON} --run="${RUN_NAME}"
done

echo ""
echo "======================================================================="
echo "Phase 1 complete (~43 runs)."
echo "Review wandb and set FINAL_* variables below before Phase 1.5."
echo "======================================================================="

# ==============================================================================
# Phase 1.5: d12 → d20 Transfer Validation (~0.75 hr, 9 runs)
# ==============================================================================
# >>> SET THESE FROM PHASE 1 RESULTS <<<
FINAL_MUON_LR=0.02
FINAL_MUON_WD=0.2
FINAL_ADAMW_LR=3e-4
FINAL_ADAMW_WD=0.1
FINAL_ADAMW_BETA2=0.95
FINAL_SOAP_LR=3e-3
FINAL_SOAP_PF=10
FINAL_SOAP_WD=0.01
FINAL_SOAP_WARMUP=0.05
FINAL_SOAP_BETA2=0.95

echo "======================================================================"
echo "Phase 1.5: d12→d20 Transfer Validation"
echo "======================================================================"

for LR_SCALE in 0.5 1.0 2.0; do
  MUON_LR=$(python -c "print(${FINAL_MUON_LR} * ${LR_SCALE})")
  ADAMW_LR=$(python -c "print(${FINAL_ADAMW_LR} * ${LR_SCALE})")
  SOAP_LR=$(python -c "print(${FINAL_SOAP_LR} * ${LR_SCALE})")

  python -m scripts.base_train --depth=${DEPTH_MAIN} --num-iterations=100 --eval-every=-1 \
    --device-batch-size=${DEVICE_BS_MAIN} \
    --optimizer-type=muon --matrix-lr=${MUON_LR} --weight-decay=${FINAL_MUON_WD} \
    --embedding-lr=0.3 \
    ${COMMON} --run="transfer_muon_scale${LR_SCALE}"

  python -m scripts.base_train --depth=${DEPTH_MAIN} --num-iterations=100 --eval-every=-1 \
    --device-batch-size=${DEVICE_BS_MAIN} \
    --optimizer-type=adamw --matrix-lr=${ADAMW_LR} --weight-decay=${FINAL_ADAMW_WD} \
    --adam-beta2=${FINAL_ADAMW_BETA2} \
    ${COMMON} --run="transfer_adamw_scale${LR_SCALE}"

  python -m scripts.base_train --depth=${DEPTH_MAIN} --num-iterations=100 --eval-every=-1 \
    --device-batch-size=${DEVICE_BS_MAIN} \
    --optimizer-type=soap --soap-lr=${SOAP_LR} --precondition-frequency=${FINAL_SOAP_PF} \
    --weight-decay=${FINAL_SOAP_WD} --warmup-ratio=${FINAL_SOAP_WARMUP} \
    --soap-beta2=${FINAL_SOAP_BETA2} \
    ${COMMON} --run="transfer_soap_scale${LR_SCALE}"
done

echo "Phase 1.5 complete. Adjust FINAL_* if transfer suggests different optimal LR."

# ==============================================================================
# Phase 2: d20 Formal Comparison (~1.5-2 hrs, 3 runs)
# ==============================================================================

echo "======================================================================"
echo "Phase 2: d20 Formal Comparison (~430M params)"
echo "======================================================================"

# 1. MuonAdamW (baseline)
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=muon \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="cmp_muon" --model-tag="cmp_muon"

# 2. PureAdamW
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=adamw \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --matrix-lr=${FINAL_ADAMW_LR} --weight-decay=${FINAL_ADAMW_WD} \
  --adam-beta2=${FINAL_ADAMW_BETA2} \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="cmp_adamw" --model-tag="cmp_adamw"

# 3. SOAP
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=soap \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --soap-lr=${FINAL_SOAP_LR} --precondition-frequency=${FINAL_SOAP_PF} \
  --weight-decay=${FINAL_SOAP_WD} --warmup-ratio=${FINAL_SOAP_WARMUP} \
  --soap-beta2=${FINAL_SOAP_BETA2} \
  --seed=${SEED} --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="cmp_soap" --model-tag="cmp_soap"

echo "Phase 2 complete."

# ==============================================================================
# Phase 3: d20 Ablation Studies (~4-6 hrs, 10 runs)
# ==============================================================================

echo "======================================================================"
echo "Phase 3: Ablation Studies (d20, ~430M params)"
echo "======================================================================"

# ---- Ablation A: Muon orthogonalization frequency (4 runs) ----
# ns_steps=1 (baseline)
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=muon \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-ns-steps=1 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="ablation_muon_ns1" --model-tag="ablation_muon_ns1"

# ns_steps=2
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=muon \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-ns-steps=2 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="ablation_muon_ns2" --model-tag="ablation_muon_ns2"

# ns_steps=5
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=muon \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-ns-steps=5 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="ablation_muon_ns5" --model-tag="ablation_muon_ns5"

# warmdown-only orthogonalization ("strong tonic hypothesis")
python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=muon \
  --device-batch-size=${DEVICE_BS_MAIN} \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-warmdown-only-ortho --muon-ns-steps=5 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
  --run="ablation_muon_warmdown_only" --model-tag="ablation_muon_warmdown_only"

# ---- Ablation B: SOAP precondition frequency (3 runs) ----
for PF in 10 50 100; do
  python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=soap \
    --device-batch-size=${DEVICE_BS_MAIN} \
    --soap-lr=${FINAL_SOAP_LR} --precondition-frequency=${PF} \
    --weight-decay=${FINAL_SOAP_WD} --warmup-ratio=${FINAL_SOAP_WARMUP} \
    --soap-beta2=${FINAL_SOAP_BETA2} \
    --seed=${SEED} --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_MAIN} \
    --run="ablation_soap_pf${PF}" --model-tag="ablation_soap_pf${PF}"
done

# ---- Ablation C: WSD warmdown ratio (3 runs) ----
for WDR in 0.25 0.5 0.75; do
  python -m scripts.base_train --depth=${DEPTH_MAIN} --optimizer-type=muon \
    --device-batch-size=${DEVICE_BS_MAIN} \
    --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
    --embedding-lr=0.3 \
    --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=${WDR} --eval-every=${EVAL_EVERY_MAIN} \
    --run="ablation_warmdown_${WDR}" --model-tag="ablation_warmdown_${WDR}"
done

echo "======================================================================"
echo "All experiments complete! Total: ~65 runs"
echo "  Phase 1:   ~43 runs  (~4.5-5.5 hr)"
echo "  Phase 1.5:   9 runs  (~0.75 hr)"
echo "  Phase 2:     3 runs  (~1.5-2 hr)"
echo "  Phase 3:    10 runs  (~4-6 hr)"
echo "======================================================================"
