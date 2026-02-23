#!/bin/bash
# ==============================================================================
# Optimizer Comparison Experiment Sweep
# Per experiment plan: experiments/optimizer_comparison.md
# ==============================================================================
set -euo pipefail

# ---- Configuration ----
export OMP_NUM_THREADS=1
DEPTH_PHASE1=8      # d8 for hyperparameter calibration
DEPTH_PHASE2=12     # d12 for formal comparison
SEED=42
EVAL_EVERY=100      # evaluation frequency (phase 1: short runs)
EVAL_EVERY_P2=250   # evaluation frequency (phase 2: full runs)
COMMON="--seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --core-metric-every=-1 --sample-every=-1 --save-every=-1"

# ==============================================================================
# Phase 1: d8 Hyperparameter Calibration (~2.5-3.5 hrs)
# ==============================================================================

echo "======================================================================"
echo "Phase 1: d8 Hyperparameter Calibration"
echo "======================================================================"

# ---- MuonAdamW calibration (12 runs: 4 LR × 3 WD) ----
# embedding_lr=0.3, unembedding_lr=0.004 fixed per §4.1
for MATRIX_LR in 0.01 0.02 0.03 0.05; do
  for WD in 0.1 0.2 0.3; do
    RUN_NAME="sweep_muon_lr${MATRIX_LR}_wd${WD}"
    echo ">> Running: ${RUN_NAME}"
    python -m scripts.base_train --depth=${DEPTH_PHASE1} --num-iterations=500 --eval-every=${EVAL_EVERY} \
      --optimizer-type=muon --matrix-lr=${MATRIX_LR} --weight-decay=${WD} \
      --embedding-lr=0.3 \
      ${COMMON} --run="${RUN_NAME}"
  done
done

# ---- AdamW calibration (30 runs: 5 LR × 3 WD × 2 beta2) ----
# beta1=0.9 fixed per §4.1
for LR in 1e-4 2e-4 3e-4 5e-4 8e-4; do
  for WD in 0.01 0.05 0.1; do
    for BETA2 in 0.95 0.99; do
      RUN_NAME="sweep_adamw_lr${LR}_wd${WD}_b2${BETA2}"
      echo ">> Running: ${RUN_NAME}"
      python -m scripts.base_train --depth=${DEPTH_PHASE1} --num-iterations=500 --eval-every=${EVAL_EVERY} \
        --optimizer-type=adamw --matrix-lr=${LR} --weight-decay=${WD} \
        --adam-beta2=${BETA2} \
        ${COMMON} --run="${RUN_NAME}"
    done
  done
done

# ---- SOAP calibration (48 runs: 4 LR × 2 beta2 × 2 PF × 2 WD × 3 warmup, reduced by fixing beta1=0.9) ----
# beta1=0.9 fixed per §4.1
for LR in 1e-3 3e-3 5e-3 1e-2; do
  for PF in 10 50; do
    for WD in 0.0 0.01; do
      for WARMUP in 0.0 0.05 0.1; do
        RUN_NAME="sweep_soap_lr${LR}_pf${PF}_wd${WD}_wu${WARMUP}"
        echo ">> Running: ${RUN_NAME}"
        python -m scripts.base_train --depth=${DEPTH_PHASE1} --num-iterations=500 --eval-every=${EVAL_EVERY} \
          --optimizer-type=soap --soap-lr=${LR} --precondition-frequency=${PF} \
          --weight-decay=${WD} --warmup-ratio=${WARMUP} \
          ${COMMON} --warmdown-ratio=0.5 \
          --run="${RUN_NAME}"
      done
    done
  done
done

# ---- SOAP beta2 search (use best LR/PF/WD from above, search beta2) ----
BEST_SOAP_LR_P1=3e-3   # placeholder — set after above sweep
BEST_SOAP_PF_P1=10     # placeholder
BEST_SOAP_WD_P1=0.01   # placeholder
BEST_SOAP_WU_P1=0.05   # placeholder
for BETA2 in 0.95 0.99; do
  RUN_NAME="sweep_soap_beta2_${BETA2}"
  echo ">> Running: ${RUN_NAME}"
  python -m scripts.base_train --depth=${DEPTH_PHASE1} --num-iterations=500 --eval-every=${EVAL_EVERY} \
    --optimizer-type=soap --soap-lr=${BEST_SOAP_LR_P1} --precondition-frequency=${BEST_SOAP_PF_P1} \
    --weight-decay=${BEST_SOAP_WD_P1} --warmup-ratio=${BEST_SOAP_WU_P1} \
    --soap-beta2=${BETA2} \
    ${COMMON} --run="${RUN_NAME}"
done

echo "Phase 1 complete. Review wandb for best hyperparameters per optimizer."
echo "Set BEST_* variables below before Phase 1.5."

# ==============================================================================
# Phase 1.5: d8 → d12 Transfer Validation (~30-45 min)
# ==============================================================================
# IMPORTANT: After Phase 1, manually set the best LRs found below.

BEST_MUON_LR=0.02
BEST_MUON_WD=0.2
BEST_ADAMW_LR=3e-4
BEST_ADAMW_WD=0.1
BEST_ADAMW_BETA2=0.95
BEST_SOAP_LR=3e-3
BEST_SOAP_PF=10
BEST_SOAP_WD=0.01
BEST_SOAP_WARMUP=0.05
BEST_SOAP_BETA2=0.95

echo "======================================================================"
echo "Phase 1.5: d8→d12 Transfer Validation"
echo "======================================================================"

for LR_SCALE in 0.5 1.0 2.0; do
  MUON_LR=$(python -c "print(${BEST_MUON_LR} * ${LR_SCALE})")
  ADAMW_LR=$(python -c "print(${BEST_ADAMW_LR} * ${LR_SCALE})")
  SOAP_LR=$(python -c "print(${BEST_SOAP_LR} * ${LR_SCALE})")

  python -m scripts.base_train --depth=${DEPTH_PHASE2} --num-iterations=100 --eval-every=-1 \
    --optimizer-type=muon --matrix-lr=${MUON_LR} --weight-decay=${BEST_MUON_WD} \
    --embedding-lr=0.3 \
    ${COMMON} --run="transfer_muon_scale${LR_SCALE}"

  python -m scripts.base_train --depth=${DEPTH_PHASE2} --num-iterations=100 --eval-every=-1 \
    --optimizer-type=adamw --matrix-lr=${ADAMW_LR} --weight-decay=${BEST_ADAMW_WD} \
    --adam-beta2=${BEST_ADAMW_BETA2} \
    ${COMMON} --run="transfer_adamw_scale${LR_SCALE}"

  python -m scripts.base_train --depth=${DEPTH_PHASE2} --num-iterations=100 --eval-every=-1 \
    --optimizer-type=soap --soap-lr=${SOAP_LR} --precondition-frequency=${BEST_SOAP_PF} \
    --weight-decay=${BEST_SOAP_WD} --warmup-ratio=${BEST_SOAP_WARMUP} \
    --soap-beta2=${BEST_SOAP_BETA2} \
    ${COMMON} --run="transfer_soap_scale${LR_SCALE}"
done

echo "Phase 1.5 complete. Set FINAL_* variables below before Phase 2."

# ==============================================================================
# Phase 2: d12 Formal Comparison (~2 hrs)
# ==============================================================================

FINAL_MUON_LR=${BEST_MUON_LR}
FINAL_MUON_WD=${BEST_MUON_WD}
FINAL_ADAMW_LR=${BEST_ADAMW_LR}
FINAL_ADAMW_WD=${BEST_ADAMW_WD}
FINAL_ADAMW_BETA2=${BEST_ADAMW_BETA2}
FINAL_SOAP_LR=${BEST_SOAP_LR}
FINAL_SOAP_PF=${BEST_SOAP_PF}
FINAL_SOAP_WD=${BEST_SOAP_WD}
FINAL_SOAP_WARMUP=${BEST_SOAP_WARMUP}
FINAL_SOAP_BETA2=${BEST_SOAP_BETA2}

echo "======================================================================"
echo "Phase 2: d12 Formal Comparison"
echo "======================================================================"

# 1. MuonAdamW (baseline)
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=muon \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="cmp_muon" --model-tag="cmp_muon"

# 2. PureAdamW
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=adamw \
  --matrix-lr=${FINAL_ADAMW_LR} --weight-decay=${FINAL_ADAMW_WD} \
  --adam-beta2=${FINAL_ADAMW_BETA2} \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="cmp_adamw" --model-tag="cmp_adamw"

# 3. SOAP
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=soap \
  --soap-lr=${FINAL_SOAP_LR} --precondition-frequency=${FINAL_SOAP_PF} \
  --weight-decay=${FINAL_SOAP_WD} --warmup-ratio=${FINAL_SOAP_WARMUP} \
  --soap-beta2=${FINAL_SOAP_BETA2} \
  --seed=${SEED} --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="cmp_soap" --model-tag="cmp_soap"

echo "Phase 2 complete."

# ==============================================================================
# Phase 3: d12 Ablation Studies
# ==============================================================================

echo "======================================================================"
echo "Phase 3: Ablation Studies"
echo "======================================================================"

# ---- Ablation A: Muon orthogonalization frequency (§6.3) ----
# Config 1: ns_steps=1 (every step, baseline)
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=muon \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-ns-steps=1 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="ablation_muon_ns1" --model-tag="ablation_muon_ns1"

# Config 2: ns_steps=2 (every 2 steps)
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=muon \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-ns-steps=2 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="ablation_muon_ns2" --model-tag="ablation_muon_ns2"

# Config 3: ns_steps=5 (every 5 steps)
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=muon \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-ns-steps=5 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="ablation_muon_ns5" --model-tag="ablation_muon_ns5"

# Config 4: warmdown-only orthogonalization ("strong tonic hypothesis")
python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=muon \
  --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
  --muon-warmdown-only-ortho --muon-ns-steps=5 --embedding-lr=0.3 \
  --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
  --run="ablation_muon_warmdown_only" --model-tag="ablation_muon_warmdown_only"

# ---- Ablation B: SOAP precondition frequency (§6.3) ----
for PF in 10 50 100; do
  python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=soap \
    --soap-lr=${FINAL_SOAP_LR} --precondition-frequency=${PF} \
    --weight-decay=${FINAL_SOAP_WD} --warmup-ratio=${FINAL_SOAP_WARMUP} \
    --soap-beta2=${FINAL_SOAP_BETA2} \
    --seed=${SEED} --warmdown-ratio=0.5 --eval-every=${EVAL_EVERY_P2} \
    --run="ablation_soap_pf${PF}" --model-tag="ablation_soap_pf${PF}"
done

# ---- Ablation C: WSD warmdown ratio (§6.3) ----
for WDR in 0.25 0.5 0.75; do
  python -m scripts.base_train --depth=${DEPTH_PHASE2} --optimizer-type=muon \
    --matrix-lr=${FINAL_MUON_LR} --weight-decay=${FINAL_MUON_WD} \
    --embedding-lr=0.3 \
    --seed=${SEED} --warmup-ratio=0.05 --warmdown-ratio=${WDR} --eval-every=${EVAL_EVERY_P2} \
    --run="ablation_warmdown_${WDR}" --model-tag="ablation_warmdown_${WDR}"
done

echo "======================================================================"
echo "All experiments complete!"
echo "======================================================================"
