#!/bin/bash

# HMM Market Regime Classifier Training Script
# This script provides easy access to different training configurations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}HMM Market Regime Classifier Training Script${NC}"
echo "============================================="

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected. Consider activating one.${NC}"
fi

# Default configuration
STATES=5
OBSERVATIONS=20
STEPS=60
STRATEGY="equal_freq"
FEATURE="sp500 high-low"
TARGET="sp500 close"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --states)
            STATES="$2"
            shift 2
            ;;
        --observations)
            OBSERVATIONS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --feature)
            FEATURE="$2"
            shift 2
            ;;
        --quick)
            STATES=3
            OBSERVATIONS=10
            STEPS=5
            echo -e "${YELLOW}Quick mode enabled: reduced parameters for fast testing${NC}"
            shift
            ;;
        --best)
            STATES=5
            OBSERVATIONS=20
            STEPS=60
            STRATEGY="equal_freq"
            FEATURE="sp500 high-low"
            echo -e "${GREEN}Best configuration mode enabled${NC}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --states N           Number of hidden states (default: 5)"
            echo "  --observations N     Number of observation bins (default: 20)"
            echo "  --steps N           Training steps (default: 60)"
            echo "  --strategy STRATEGY  Discretization strategy (default: equal_freq)"
            echo "  --feature FEATURE   Feature column (default: 'sp500 high-low')"
            echo "  --quick             Quick test with reduced parameters"
            echo "  --best              Use best known configuration"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Configuration:${NC}"
echo "  States: $STATES"
echo "  Observations: $OBSERVATIONS"
echo "  Steps: $STEPS"
echo "  Strategy: $STRATEGY"
echo "  Feature: $FEATURE"
echo "  Target: $TARGET"
echo ""

# Run training
echo -e "${GREEN}Starting training...${NC}"
python -m src.training.train \
    --mode classification \
    --states $STATES \
    --observations $OBSERVATIONS \
    --steps $STEPS \
    --discr_strategy $STRATEGY \
    --direct_states \
    --feature "$FEATURE" \
    --target "$TARGET" \
    --final_test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo "Results saved in src/results/"
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi 