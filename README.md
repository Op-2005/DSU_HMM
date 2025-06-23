# HMM Market Regime Classifier 📈

A sophisticated Hidden Markov Model implementation for financial market regime classification using PyTorch. This project identifies distinct market regimes (bull, bear, and sideways markets) from financial time series data with high accuracy and interpretability.

## 🌟 Features

- **Custom HMM Implementation**: PyTorch-based Hidden Markov Model with optimized Viterbi decoding and Baum-Welch EM training
- **Financial Data Processing**: Specialized data loaders with multiple discretization strategies (equal frequency, equal width, K-means)
- **Market Regime Classification**: Automatically identifies bull, bear, and mixed market conditions
- **Interactive Dashboard**: Streamlit web application for real-time model visualization and analysis
- **Advanced Model Architecture**: Structured emission matrices for improved regime separation
- **Comprehensive Evaluation**: Detailed performance metrics, confusion matrices, and state interpretations
- **Production Ready**: Full test suite, type hints, and production deployment capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hmm-market-regime-classifier.git
cd DSU_HMM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

#### 1. Train a Model

```bash
# Quick training with default parameters
python -m src.training.train --mode classification --states 5 --observations 20

# Or use the convenient training script
./scripts/run_training.sh --best
```

#### 2. Launch the Dashboard

```bash
# Prepare dashboard data
python -m src.web_app.prepare_data

# Launch interactive dashboard
streamlit run src/web_app/app.py
```

#### 3. Visualize Results

```python
from src.visualization.visualize_model import create_all_visualizations

# Generate comprehensive visualizations
create_all_visualizations(
    model_path="src/results/optimized_hmm_classification_model.pt",
    data_path="src/data/financial_data.csv"
)
```

## 📊 Model Performance

The current best model achieves:

- **Accuracy**: 66.12%
- **Precision**: 70.83%
- **Recall**: 71.33%
- **F1 Score**: 71.08%

### State Interpretations

| State | Type | Bull Ratio | Mean Return | Regime Description |
|-------|------|------------|-------------|-------------------|
| 0 | Bull Market | 0.76 | 3.55 | Strong upward trends |
| 1 | Mixed Market | 0.49 | 4.52 | Neutral/sideways movement |
| 2 | Mixed Market | 0.63 | 6.03 | Mild bullish bias |
| 3 | Mixed Market | 0.43 | 7.43 | Mild bearish bias |
| 4 | Bear Market | 0.14 | 12.44 | Strong downward trends |

## 🏗️ Project Structure

```
DSU_HMM/
├── src/
│   ├── data/                    # Data processing and loading
│   │   ├── data_processor.py    # Financial data loader and preprocessing
│   │   └── financial_data.csv   # Sample financial dataset
│   ├── models/                  # Core HMM implementation
│   │   ├── hmm_model.py        # Main HMM class with Viterbi & Baum-Welch
│   │   └── hyperparameter_test.py # Model optimization and testing
│   ├── training/                # Training scripts and utilities
│   │   └── train.py            # Main training script with CLI interface
│   ├── visualization/           # Model visualization and analysis
│   │   ├── visualize_model.py  # Core visualization functions
│   │   ├── view_model.py       # Model analysis utilities
│   │   └── model_visualizations/ # Generated plots and charts
│   ├── web_app/                # Interactive dashboard
│   │   ├── app.py              # Streamlit dashboard application
│   │   └── prepare_data.py     # Dashboard data preparation
│   └── results/                # Model outputs and reports
│       ├── final_model_results.md
│       ├── optimized_hmm_classification_model.pt
│       └── structured_emission_model_results.json
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── scripts/                    # Utility scripts
│   └── run_training.sh         # Training automation script
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── setup.py                   # Package setup
```

## 🔧 Advanced Usage

### Custom Training Configuration

```bash
python -m src.training.train \
    --mode classification \
    --states 6 \
    --observations 25 \
    --steps 100 \
    --discr_strategy kmeans \
    --direct_states \
    --feature "sp500 volume" \
    --target "sp500 close" \
    --class_threshold 0.3
```

### Training Script Options

```bash
# Quick test with reduced parameters
./scripts/run_training.sh --quick

# Best known configuration
./scripts/run_training.sh --best

# Custom configuration
./scripts/run_training.sh --states 6 --observations 30 --steps 80
```

### Using the Python API

```python
from src.models.hmm_model import HiddenMarkovModel
from src.data.data_processor import FinancialDataLoader, discretize_data

# Load and prepare data
loader = FinancialDataLoader(
    file_path="src/data/financial_data.csv",
    target_column="sp500 close",
    features=["sp500 high-low"]
)

# Discretize features
X_discrete = discretize_data(loader.X, num_bins=20, strategy='equal_freq')

# Initialize and train HMM
hmm = HiddenMarkovModel(T, E, T0, device='cpu')
T0, T, E, converged = hmm.Baum_Welch_EM(X_discrete)

# Perform inference
states, probabilities = hmm.viterbi_inference(X_discrete)
```

## 📈 Dashboard Features

The interactive Streamlit dashboard provides:

- **Real-time Model Performance**: Live accuracy, precision, recall metrics
- **State Analysis**: Detailed regime interpretations and transition matrices
- **Time Series Visualization**: Interactive plots with regime highlighting
- **Confusion Matrix**: Model classification performance breakdown
- **Parameter Exploration**: Dynamic model configuration and results

Access the dashboard at `http://localhost:8501` after running:

```bash
streamlit run src/web_app/app.py
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/unit/test_hmm_model.py
```

## 📚 API Reference

### Core Classes

#### `HiddenMarkovModel`
Main HMM implementation with PyTorch backend.

```python
model = HiddenMarkovModel(
    T,           # Transition matrix (S x S)
    E,           # Emission matrix (S x O)
    T0,          # Initial state distribution (S,)
    device='cpu', # Computation device
    epsilon=0.001, # Convergence threshold
    maxStep=20    # Maximum EM iterations
)
```

#### `FinancialDataLoader`
Specialized data loader for financial time series.

```python
loader = FinancialDataLoader(
    file_path,     # Path to CSV file
    target_column, # Target variable name
    features,      # List of feature column names
    normalize=True # Whether to normalize features
)
```

### Key Methods

- **`viterbi_inference(observations)`**: Find most likely state sequence
- **`Baum_Welch_EM(observations)`**: Train model parameters using EM algorithm
- **`evaluate(observations, mode='classification')`**: Evaluate model performance
- **`save_model(filepath)`** / **`load_model(filepath)`**: Model persistence

## 🔬 Research & Applications

### Academic Applications
- Market regime identification and analysis
- Financial time series modeling
- Behavioral finance research
- Risk management and portfolio optimization

### Industry Applications
- Algorithmic trading strategy development
- Risk-adjusted portfolio allocation
- Market timing and regime-based investment
- Financial advisory and robo-advisor integration

## 🛠️ Development

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Quality

The project uses:
- **Black**: Code formatting
- **MyPy**: Static type checking
- **Flake8**: Linting
- **Pytest**: Testing framework

Run quality checks:

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/

# All checks
pre-commit run --all-files
```

## 🚀 Future Enhancements

See [`NEXT_STEPS.md`](NEXT_STEPS.md) for detailed roadmap including:

- **Advanced Model Architectures**: Hierarchical HMMs, time-varying parameters
- **Feature Engineering**: Technical indicators, alternative data sources
- **Real-time Integration**: Live data feeds and streaming inference
- **Production Deployment**: Cloud deployment, API development
- **Research Extensions**: Deep learning integration, reinforcement learning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/hmm-market-regime-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hmm-market-regime-classifier/discussions)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- Built with PyTorch, Streamlit, and the Python scientific computing stack
- Inspired by classical Hidden Markov Model literature and modern financial ML research
- Special thanks to the open-source community for excellent tools and libraries

---

**⭐ Star this repository if you find it useful!** 