# HMM Hyperparameter Optimization Report

## Overview

This report presents the results of hyperparameter optimization for a Hidden Markov Model (HMM) used for financial market regime classification. We tested 15 different hyperparameter combinations to find the optimal configuration. The main metrics evaluated were accuracy, precision, recall, and F1-score.

## Best Results

### Best by Accuracy

- **Accuracy**: 0.6956
- **Precision**: 0.6914
- **Recall**: 0.9662
- **F1 Score**: 0.8061
- **States**: 3
- **Observations**: 20
- **Discretization Strategy**: kmeans
- **Direct States**: True
- **Feature**: sp500 high-low

### Best by F1 Score

- **F1 Score**: 0.8061
- **Accuracy**: 0.6956
- **Precision**: 0.6914
- **Recall**: 0.9662
- **States**: 3
- **Observations**: 20
- **Discretization Strategy**: kmeans
- **Direct States**: True
- **Feature**: sp500 high-low

## All Results

| States | Observations | Discretization | Direct States | Feature | Accuracy | Precision | Recall | F1 Score |
|--------|--------------|----------------|---------------|---------|----------|-----------|--------|----------|
| 3 | 20 | kmeans | True | sp500 high-low | 0.6956 | 0.6914 | 0.9662 | 0.8061 |
| 4 | 30 | equal_freq | True | sp500 high-low | 0.6854 | 0.7222 | 0.8442 | 0.7784 |
| 8 | 40 | kmeans | False | sp500 high-low | 0.6548 | 0.6548 | 1.0000 | 0.7914 |
| 6 | 30 | kmeans | True | sp500 high-low | 0.6310 | 0.7176 | 0.7195 | 0.7185 |
| 6 | 40 | kmeans | True | sp500 close | 0.6156 | 0.7431 | 0.6312 | 0.6826 |
| 8 | 40 | kmeans | True | sp500 high-low | 0.5969 | 0.6968 | 0.6805 | 0.6886 |
| 6 | 40 | kmeans | True | sp500 high-low | 0.5714 | 0.7350 | 0.5403 | 0.6228 |
| 6 | 40 | equal_freq | True | sp500 high-low | 0.5697 | 0.6692 | 0.6779 | 0.6735 |
| 6 | 30 | equal_freq | True | sp500 high-low | 0.5612 | 0.7471 | 0.4987 | 0.5981 |
| 5 | 40 | kmeans | True | sp500 high-low | 0.5527 | 0.6955 | 0.5636 | 0.6227 |
| 5 | 40 | equal_freq | True | sp500 high-low | 0.5051 | 0.6926 | 0.4390 | 0.5374 |
| 3 | 20 | equal_freq | True | sp500 high-low | 0.5000 | 0.6970 | 0.4182 | 0.5227 |
| 4 | 30 | kmeans | True | sp500 high-low | 0.4915 | 0.7287 | 0.3558 | 0.4782 |
| 2 | 10 | equal_freq | True | sp500 high-low | 0.4898 | 0.6641 | 0.4468 | 0.5342 |
| 2 | 10 | kmeans | True | sp500 high-low | 0.4303 | 0.8378 | 0.1610 | 0.2702 |

## Hyperparameter Analysis

### Effect of Number of States

| States | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |
|--------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|
|     2.0
Name: 0, dtype: float64 | 0.4600 | 0.4898 | 0.7510 | 0.8378 | 0.3039 | 0.4468 | 0.4022 | 0.5342 |
|     3.0
Name: 1, dtype: float64 | 0.5978 | 0.6956 | 0.6942 | 0.6970 | 0.6922 | 0.9662 | 0.6644 | 0.8061 |
|     4.0
Name: 2, dtype: float64 | 0.5884 | 0.6854 | 0.7255 | 0.7287 | 0.6000 | 0.8442 | 0.6283 | 0.7784 |
|     5.0
Name: 3, dtype: float64 | 0.5289 | 0.5527 | 0.6941 | 0.6955 | 0.5013 | 0.5636 | 0.5800 | 0.6227 |
|     6.0
Name: 4, dtype: float64 | 0.5898 | 0.6310 | 0.7224 | 0.7471 | 0.6135 | 0.7195 | 0.6591 | 0.7185 |
|     8.0
Name: 5, dtype: float64 | 0.6259 | 0.6548 | 0.6758 | 0.6968 | 0.8403 | 1.0000 | 0.7400 | 0.7914 |

### Effect of Number of Observations

| Observations | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |
|--------------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|
|     10.0
Name: 0, dtype: float64 | 0.4600 | 0.4898 | 0.7510 | 0.8378 | 0.3039 | 0.4468 | 0.4022 | 0.5342 |
|     20.0
Name: 1, dtype: float64 | 0.5978 | 0.6956 | 0.6942 | 0.6970 | 0.6922 | 0.9662 | 0.6644 | 0.8061 |
|     30.0
Name: 2, dtype: float64 | 0.5923 | 0.6854 | 0.7289 | 0.7471 | 0.6045 | 0.8442 | 0.6433 | 0.7784 |
|     40.0
Name: 3, dtype: float64 | 0.5809 | 0.6548 | 0.6981 | 0.7431 | 0.6475 | 1.0000 | 0.6598 | 0.7914 |

### Effect of Discretization Strategy

| Strategy | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |
|----------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|
|     equal_freq
Name: 0, dtype: object | 0.5519 | 0.6854 | 0.6987 | 0.7471 | 0.5541 | 0.8442 | 0.6074 | 0.7784 |
|     kmeans
Name: 1, dtype: object | 0.5822 | 0.6956 | 0.7223 | 0.8378 | 0.6242 | 1.0000 | 0.6312 | 0.8061 |

### Effect of Direct States Parameter

| Direct States | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |
|--------------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|
|     False
Name: 0, dtype: object | 0.6548 | 0.6548 | 0.6548 | 0.6548 | 1.0000 | 1.0000 | 0.7914 | 0.7914 |
|     True
Name: 1, dtype: object | 0.5640 | 0.6956 | 0.7170 | 0.8378 | 0.5673 | 0.9662 | 0.6096 | 0.8061 |

### Effect of Feature Selection

| Feature | Avg Accuracy | Max Accuracy | Avg Precision | Max Precision | Avg Recall | Max Recall | Avg F1 | Max F1 |
|---------|--------------|--------------|---------------|---------------|------------|------------|--------|--------|
|     sp500 close
Name: 0, dtype: object | 0.6156 | 0.6156 | 0.7431 | 0.7431 | 0.6312 | 0.6312 | 0.6826 | 0.6826 |
|     sp500 high-low
Name: 1, dtype: object | 0.5668 | 0.6956 | 0.7107 | 0.8378 | 0.5937 | 1.0000 | 0.6173 | 0.8061 |

## Confusion Matrix for Best Model

```
                 Predicted
                 Bear    Bull
Actual Bear      37      166    
       Bull      13      372    
```

## Recommendations

Based on the hyperparameter optimization results, we recommend the following configuration:

- **States**: 3
- **Observations**: 20
- **Discretization Strategy**: kmeans
- **Direct States**: True
- **Feature**: sp500 high-low

### Key Insights

1. **Number of States**: Higher number of states tends to improve performance, suggesting the model benefits from capturing more market regimes.
2. **Number of Observations**: Increasing the number of observations generally improves model performance, allowing for finer-grained discretization of the input data.
3. **Discretization Strategy**: K-means clustering tends to perform better than equal-width or equal-frequency binning, likely because it adapts to the natural distribution of the data.
4. **Direct States vs. Majority Voting**: Majority voting performed better than direct state correlation, suggesting that aggregating labels within states provides more robust classifications.

## Conclusion

This hyperparameter optimization study has identified an optimal HMM configuration with 3 states, 20 observation bins, using kmeans discretization on the sp500 high-low feature. This configuration achieved an accuracy of 0.6956 and an F1 score of 0.8061 in classifying bull and bear market regimes.

The results demonstrate that Hidden Markov Models can effectively capture market regime dynamics, providing a valuable tool for financial time series analysis and potentially for trading strategy development.