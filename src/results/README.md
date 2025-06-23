# HMM Financial Market Regime Classifier

A sophisticated Hidden Markov Model (HMM) implementation for classifying financial market regimes (bull vs bear markets) using time series analysis and structured emission matrices.

## 🎯 Project Overview

This project implements a state-of-the-art Hidden Markov Model to identify distinct market regimes in financial time series data. The model automatically discovers hidden states that correspond to different market conditions (bull markets, bear markets, and transitional periods) based on price and volatility patterns.

### Key Features

- **Structured Emission Matrix**: Domain knowledge-informed initialization for better regime separation
- **Multiple Discretization Strategies**: Equal frequency, equal width, and K-means clustering
- **Direct State Mapping**: Advanced classification approach using state correlations
- **Interactive Dashboard**: Streamlit-based web interface for model visualization
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, and F1-score

## 🏗️ Architecture

### Core Components

```
src/
├── models/
│   ├── hmm_model.py          # Core HMM implementation with Viterbi & Forward-Backward
│   └── hyperparameter_test.py # Grid search and model optimization
├── data/
│   ├── data_processor.py     # Financial data loading and preprocessing
│   └── financial_data.csv    # S&P 500 historical data
├── training/
│   └── train.py             # Model training pipeline
├── visualization/
│   ├── visualize_*.py       # Various plotting utilities
│   └── model_visualizations/ # Generated plots and charts
├── web_app/
│   ├── app.py              # Streamlit dashboard
│   ├── prepare_data.py     # Dashboard data preparation
│   └── dashboard_data.json # Preprocessed dashboard data
└── results/
    ├── *.json              # Model results and metrics
    ├── *.md               # Generated reports
    └── *.pt               # Saved model weights
```

## 🔬 Technical Deep Dive

### Hidden Markov Model Implementation

The HMM implementation (`hmm_model.py`) provides:

1. **Forward-Backward Algorithm**: Efficient computation of state probabilities
2. **Viterbi Algorithm**: Most likely state sequence inference
3. **Baum-Welch EM**: Parameter estimation through iterative optimization
4. **Structured Initialization**: Domain-informed parameter initialization

### Market Regime Detection

The model identifies distinct market states:

- **State 0 (Bull Market)**: High bull ratio (0.76), low volatility
- **State 1-3 (Mixed/Transitional)**: Varying bull ratios (0.43-0.63)
- **State 4 (Bear Market)**: Low bull ratio (0.14), high volatility

### Data Processing Pipeline

1. **Feature Engineering**: Log returns, volatility measures, high-low ranges
2. **Discretization**: Converting continuous features to discrete observations
3. **Regime Labeling**: Bull/bear classification based on return thresholds
4. **Normalization**: Statistical standardization for stable training

## 📊 Performance Metrics

### Current Best Model Configuration

```python
{
    "states": 5,
    "observations": 20,
    "discretization": "equal_freq",
    "classification_threshold": 0.4,
    "training_steps": 60,
    "feature": "sp500 high-low"
}
```

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 66.12% |
| **Precision** | 70.83% |
| **Recall** | 71.33% |
| **F1 Score** | 71.08% |

### Confusion Matrix
```
                 Predicted
                 Bear    Bull
Actual Bear      180     126
       Bull      123     306
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DSU_HMM

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

```bash
# Basic training
python src/training/train.py --mode classification --states 5 --observations 20

# Advanced training with custom parameters
python src/training/train.py \
    --mode classification \
    --states 6 \
    --observations 40 \
    --discr_strategy kmeans \
    --direct_states \
    --feature "sp500 high-low" \
    --target "sp500 close" \
    --steps 40
```

### Hyperparameter Optimization

```bash
# Run comprehensive grid search
python src/models/hyperparameter_test.py

# Quick single test
python src/models/hyperparameter_test.py --single_test
```

### Web Dashboard

```bash
# Launch interactive dashboard
cd src/web_app
streamlit run app.py
```

## 📈 Model Interpretability

### State Interpretations

The model learns meaningful market regimes:

- **Bull Markets**: Characterized by consistent positive returns and lower volatility
- **Bear Markets**: High volatility periods with predominantly negative returns
- **Transitional States**: Mixed periods that often precede regime changes

### Transition Dynamics

The transition matrix reveals:
- High self-transition probabilities (regime persistence)
- Symmetric transition patterns between bull and bear states
- Intermediate states acting as buffers during regime changes

## 🔧 Configuration

### Key Parameters

- **`num_states`**: Number of hidden market regimes (typically 3-7)
- **`num_observations`**: Discretization bins for continuous features
- **`max_steps`**: Maximum EM iterations for convergence
- **`classification_threshold`**: Bull/bear classification boundary
- **`discretization_strategy`**: Method for converting continuous to discrete data

### Feature Engineering Options

- **Price-based**: Close prices, high-low ranges, open-close differences
- **Return-based**: Log returns, rolling volatility, momentum indicators
- **Technical**: Moving averages, RSI, Bollinger Bands (extensible)

## 📋 Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework for tensor operations
- **NumPy/Pandas**: Numerical computing and data manipulation
- **Scikit-learn**: Machine learning utilities and metrics
- **Matplotlib/Plotly**: Visualization and charting

### Web Interface

- **Streamlit**: Interactive dashboard framework
- **Plotly**: Interactive plotting library

## 🧪 Testing and Validation

### Model Validation

- **Time-series split**: Chronological train/validation/test splits
- **Cross-validation**: Rolling window validation for temporal data
- **Regime consistency**: State interpretation validation across time periods

### Performance Monitoring

- **Convergence tracking**: EM algorithm convergence monitoring
- **State stability**: Transition matrix eigenvalue analysis
- **Out-of-sample testing**: Forward-looking validation

## 📚 Research Background

### Theoretical Foundation

This implementation is based on:

1. **Hidden Markov Models** (Rabiner, 1989): Probabilistic sequence modeling
2. **Regime-Switching Models** (Hamilton, 1989): Economic time series analysis
3. **Market Microstructure** (O'Hara, 1995): High-frequency trading patterns

### Academic References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition
- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series
- Kim, C. J., & Nelson, C. R. (1999). State-space models with regime switching

## 🛠️ Development

### Code Structure

The codebase follows software engineering best practices:

- **Modular design**: Separated concerns for data, models, and visualization
- **Type hints**: Enhanced code readability and IDE support
- **Documentation**: Comprehensive docstrings and comments
- **Error handling**: Robust exception management throughout

### Testing Framework

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Performance benchmarking
python scripts/benchmark.py
```

## 🔮 Future Enhancements

### Model Improvements

1. **Multi-asset HMMs**: Portfolio-level regime detection
2. **Time-varying parameters**: Adaptive model parameters
3. **Hybrid approaches**: Combining HMMs with neural networks
4. **Alternative features**: Sentiment analysis, news impact

### Technical Enhancements

1. **GPU acceleration**: CUDA support for large-scale training
2. **Real-time inference**: Live market regime classification
3. **API integration**: Bloomberg/Yahoo Finance data feeds
4. **Cloud deployment**: Scalable model serving infrastructure

### Research Directions

1. **Regime prediction**: Forecasting regime transitions
2. **Portfolio optimization**: Regime-aware asset allocation
3. **Risk management**: Dynamic hedging strategies
4. **Alternative assets**: Cryptocurrency and commodity markets

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Issues**: Please use GitHub Issues for bug reports and feature requests
- **Discussions**: Join our GitHub Discussions for general questions
- **Email**: [Contact information]

---

**Note**: This model is for research and educational purposes. Always consult with financial professionals before making investment decisions based on model outputs. 