#!/bin/bash
# LoRA-based Federated Training for DiSan - V2 (Stronger Privacy)
#
# Key improvement: Increased privacy parameters to match baseline exp6
# - grl_lambda: 0.3 -> 0.5 (stronger gradient reversal)
# - server_adv_steps: 30 -> 100 (stronger adversary)
# - lambda_proto_adv: 0.3 -> 1.0 (stronger prototype adversarial loss)
# - num_rounds: 10 -> 12 (more training)
#
# Expected: Role probe accuracy closer to random baseline (14.29%)

set -euo pipefail

cd "$(dirname "$0")/.."

CLIENTS="Client_1_CorporateBank Client_2_AssetManager Client_3_FinTechPay Client_4_CorpGroup Client_5_MarketForecaster Client_6_ComplianceConsult Client_7_SupplierCo"
CUDA_DEVICES="0,1,2,3,4"
DEVICES="0 1 2 3 4"
DATA_ROOT="./data"
OUTPUT_DIR="./checkpoints/fed_lora_v2"
LOG_DIR="./logs"

# Training params
NUM_ROUNDS=12
LOCAL_STEPS=300
LR=2e-4
BATCH_SIZE=4

# LoRA params
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1

# Privacy params (STRONGER than v1)
GRL_LAMBDA=0.5              # was 0.3
SERVER_ADV_STEPS=100        # was 30
LAMBDA_PROTO_ADV=1.0        # was 0.3
ADV_LOSS_CLAMP=10.0         # was 5.0

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "LoRA Federated Training V2 (Stronger Privacy)"
echo "========================================"
echo "Clients: $CLIENTS"
echo "Output: $OUTPUT_DIR"
echo ""
echo "LoRA configuration:"
echo "  lora_r: $LORA_R"
echo "  lora_alpha: $LORA_ALPHA"
echo "  lora_dropout: $LORA_DROPOUT"
echo ""
echo "Privacy configuration (STRONGER):"
echo "  grl_lambda: $GRL_LAMBDA (was 0.3)"
echo "  server_adv_steps: $SERVER_ADV_STEPS (was 30)"
echo "  lambda_proto_adv: $LAMBDA_PROTO_ADV (was 0.3)"
echo "  adv_loss_clamp: $ADV_LOSS_CLAMP (was 5.0)"
echo ""
echo "Expected improvement: Role probe accuracy closer to random (14.29%)"
echo ""

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

python -m fed_lightweight.train_federated_parallel \
    --clients $CLIENTS \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --devices $DEVICES \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --local_steps $LOCAL_STEPS \
    --num_rounds $NUM_ROUNDS \
    --lambda_orth 0.2 \
    --grl_lambda $GRL_LAMBDA \
    --fedprox_mu 0.1 \
    --adv_loss_clamp $ADV_LOSS_CLAMP \
    --server_adv_steps $SERVER_ADV_STEPS \
    --residual_mode full \
    --enable_two_stream \
    --lambda_priv_start 0.5 \
    --lambda_priv_end 3.0 \
    --lambda_priv_warmup_steps 100 \
    --lambda_proto_align 2.0 \
    --proto_align_warmup_steps 30 \
    --proto_align_start_round 2 \
    --lambda_proto_adv $LAMBDA_PROTO_ADV \
    --proto_adv_warmup_steps 100 \
    --use_spherical_align \
    --lambda_sphere_direction 3.0 \
    --lambda_sphere_dispersion 0.2 \
    --proto_noise_scale 0.01 \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    2>&1 | tee "$LOG_DIR/fed_lora_v2_${TIMESTAMP}.log"

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Log: $LOG_DIR/fed_lora_v2_${TIMESTAMP}.log"
echo "========================================"
echo ""
echo "Next: Run attack experiments to verify improvement:"
echo "  python -m attack_light.exp1_representation_probe --checkpoint_dir $OUTPUT_DIR --use_lora"
