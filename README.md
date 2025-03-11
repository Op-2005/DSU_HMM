# Hidden Markov Model for Financial Market Regime Detection

## Project Overview
This project implements a Hidden Markov Model (HMM) to detect and classify financial market regimes (bull and bear markets) based on financial time series data.

## Finalized Model
The most recent and optimized version uses a **structured emission matrix** to improve market regime separation, with the following characteristics:
- 5 states: 2 bull market states, 2 mixed market states, and 1 clear bear market state
- 20 discretized observations using equal-frequency binning
- Uses "sp500 high-low" as the primary feature
- Classification threshold of 0.45 to better identify bear markets

## Performance Metrics
The finalized model achieved:
- **Accuracy**: 0.6190
- **Precision**: 0.7492 (higher than baseline)
- **Recall**: 0.5221
- **F1 Score**: 0.6154

While accuracy and recall are lower than the baseline model, precision improved by +6.47%. This indicates a trade-off where the model is more conservative but more accurate when it does predict a bull market.

## Key Files

### Code Files
- `hyperparameter_test.py` - Contains the finalized model implementation with structured emission matrix
- `hmm_model.py` - The HMM implementation with fixed emission matrix indexing (shape: states × observations)
- `data_processor.py` - Data loading and preprocessing utilities

### Results and Models
- `structured_emission_model_report.md` - **IMPORTANT**: Detailed report of the finalized model performance
- `structured_emission_model_results.json` - Raw metrics data
- `optimized_hmm_classification_model.pt` - Saved model weights

## Key Code Improvements
1. Fixed emission matrix shape to be (num_states, num_observations)
2. Implemented structured emission matrix initialization for better regime separation
3. Added robust error handling throughout the codebase
4. Added detailed state interpretations to identify true bear markets

## Running the Model
```
python hyperparameter_test.py
```

## Future Work
1. Add additional features like volatility measures
2. Implement regime-specific feature engineering
3. Explore time-varying transition probabilities
4. Develop and backtest trading strategies based on identified regimes 
