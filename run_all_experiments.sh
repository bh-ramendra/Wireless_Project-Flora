#!/bin/bash
# run_all_experiments.sh
# Runs all FL methods across all required Dirichlet alpha values.
# Edit NUM_ROUNDS to reduce run time if needed.

set -e
cd "$(dirname "$0")"

ALPHAS=(0.5 1.0)
METHODS=(fedavg flora fedsb)
NUM_CLIENTS=10
NUM_ROUNDS=20

echo "======================================================"
echo "  Federated Learning Experiments — FL with LLMs"
echo "======================================================"

for METHOD in "${METHODS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        echo ""
        echo ">>> Running: method=$METHOD  alpha=$ALPHA  clients=$NUM_CLIENTS"
        
        # Write a temp config with the right alpha
        TMP_CONFIG="configs/tmp_${METHOD}_alpha${ALPHA}.yaml"
        cp "configs/${METHOD}.yaml" "$TMP_CONFIG"
        sed -i "s/dirichlet_alpha:.*/dirichlet_alpha: ${ALPHA}/" "$TMP_CONFIG"
        sed -i "s/num_clients:.*/num_clients: ${NUM_CLIENTS}/"   "$TMP_CONFIG"
        sed -i "s/num_rounds:.*/num_rounds: ${NUM_ROUNDS}/"       "$TMP_CONFIG"

        python src/server.py --config "$TMP_CONFIG"
        rm "$TMP_CONFIG"
    done
done

echo ""
echo "======================================================"
echo "  All experiments done! Generating plots..."
echo "======================================================"

python src/plot_results.py --results_dir results/ --num_clients $NUM_CLIENTS

echo ""
echo "  Plots saved in results/"
echo "======================================================"
