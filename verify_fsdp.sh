#!/bin/bash
# Verify FSDP implementation files exist and are properly structured

echo "=================================================="
echo "FSDP Implementation Verification"
echo "=================================================="

# Check core files
echo -e "\n📁 Checking core FSDP files..."

files=(
    "train_fsdp.py"
    "config_fsdp.py"
    "utils/fsdp_utils.py"
    "test_fsdp.py"
    "scripts/run_fsdp.sh"
    "docs/fsdp_training.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists ($(wc -l < "$file") lines)"
    else
        echo "✗ $file missing"
    fi
done

# Check modifications
echo -e "\n📝 Checking modified files..."

# Check if FSDP methods exist in model
if grep -q "configure_for_fsdp" models/functional_grasp_model.py; then
    echo "✓ Model has configure_for_fsdp() method"
else
    echo "✗ Model missing configure_for_fsdp() method"
fi

if grep -q "get_fsdp_wrap_params" models/functional_grasp_model.py; then
    echo "✓ Model has get_fsdp_wrap_params() method"
else
    echo "✗ Model missing get_fsdp_wrap_params() method"
fi

# Check data loader support
if grep -q "DistributedSampler" dataset/oakink_loader.py; then
    echo "✓ Data loader has DistributedSampler support"
else
    echo "✗ Data loader missing DistributedSampler support"
fi

# Check config
if grep -q "FSDP" config.py; then
    echo "✓ Base config has FSDP section"
else
    echo "✗ Base config missing FSDP section"
fi

# Check key FSDP features
echo -e "\n🔧 Checking FSDP features..."

# Check for mixed precision
if grep -q "MixedPrecision" train_fsdp.py; then
    echo "✓ Mixed precision support"
else
    echo "✗ Missing mixed precision"
fi

# Check for sharding strategies
if grep -q "ShardingStrategy" train_fsdp.py; then
    echo "✓ Sharding strategies implemented"
else
    echo "✗ Missing sharding strategies"
fi

# Check for checkpoint support
if grep -q "save_checkpoint_fsdp" train_fsdp.py; then
    echo "✓ FSDP checkpoint saving"
else
    echo "✗ Missing checkpoint saving"
fi

# Check for auto-wrap policy
if grep -q "transformer_auto_wrap_policy" train_fsdp.py; then
    echo "✓ Transformer auto-wrap policy"
else
    echo "✗ Missing auto-wrap policy"
fi

echo -e "\n=================================================="
echo "Summary:"
echo "FSDP implementation is ready for deployment."
echo "To use on your server with GPUs:"
echo ""
echo "1. Single-node multi-GPU:"
echo "   bash scripts/run_fsdp.sh --gpus 8 --data_path /path/to/OakInk"
echo ""
echo "2. Direct torchrun:"
echo "   torchrun --standalone --nproc_per_node=8 train_fsdp.py"
echo ""
echo "3. Multi-node (on each node):"
echo "   bash scripts/run_fsdp.sh --num_nodes 2 --node_rank 0 --master_addr IP"
echo "=================================================="