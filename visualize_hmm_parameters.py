#!/usr/bin/env python
"""
Script to visualize HMM model parameters and results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os
from pathlib import Path

# Set style for all plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create directory for visualizations
output_dir = Path('model_visualizations')
output_dir.mkdir(exist_ok=True)

def plot_transition_matrix(transition_matrix, title="Transition Matrix"):
    """Plot the transition matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
               xticklabels=[f"State {i}" for i in range(len(transition_matrix))],
               yticklabels=[f"State {i}" for i in range(len(transition_matrix))])
    plt.title(title, fontsize=16)
    plt.xlabel("To State", fontsize=14)
    plt.ylabel("From State", fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'transition_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Transition matrix plot saved to {output_dir / 'transition_matrix.png'}")

def plot_emission_matrix(emission_matrix, title="Emission Matrix"):
    """Plot the emission matrix as a heatmap"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(emission_matrix, annot=False, cmap="viridis",
               xticklabels=[f"Obs {i}" for i in range(emission_matrix.shape[1])],
               yticklabels=[f"State {i}" for i in range(emission_matrix.shape[0])])
    plt.title(title, fontsize=16)
    plt.xlabel("Observation", fontsize=14)
    plt.ylabel("State", fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'emission_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Emission matrix plot saved to {output_dir / 'emission_matrix.png'}")

def plot_initial_distribution(initial_dist, title="Initial State Distribution"):
    """Plot the initial state distribution as a bar chart"""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(initial_dist)), initial_dist, color='skyblue')
    plt.xticks(range(len(initial_dist)), [f"State {i}" for i in range(len(initial_dist))])
    plt.title(title, fontsize=16)
    plt.xlabel("State", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'initial_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Initial distribution plot saved to {output_dir / 'initial_distribution.png'}")

def plot_performance_comparison():
    """Plot performance comparison between different models"""
    try:
        # Define the models and their metrics
        models = ['Baseline Model', 'Previous Structured Model', 'Improved Model']
        metrics = {
            'Accuracy': [0.6599, 0.6190, 0.6612],
            'Precision': [0.6845, 0.7492, 0.7083],
            'Recall': [0.7739, 0.5221, 0.7133],
            'F1 Score': [0.7265, 0.6154, 0.7108]
        }
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(metrics, index=models)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        ax = df.plot(kind='bar', width=0.8, figsize=(14, 8))
        plt.title('Performance Comparison Between Models', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(title='Metric', fontsize=12)
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison saved to {output_dir / 'model_performance_comparison.png'}")
        
        return True
    except Exception as e:
        print(f"Error plotting performance comparison: {e}")
        return False

def plot_state_interpretations():
    """Plot state interpretations from the model results"""
    try:
        # Load model results
        with open('structured_emission_model_results.json', 'r') as f:
            results = json.load(f)
        
        # Extract state interpretations
        states = []
        bull_ratios = []
        mean_returns = []
        std_devs = []
        
        for i, state_info in enumerate(results.get('state_interpretations', {}).values()):
            states.append(f"State {i}")
            bull_ratios.append(state_info.get('bull_ratio', 0))
            mean_returns.append(state_info.get('mean_return', 0))
            std_devs.append(state_info.get('std_deviation', 0))
        
        # Create a DataFrame
        df = pd.DataFrame({
            'State': states,
            'Bull Ratio': bull_ratios,
            'Mean Return': mean_returns,
            'Std Deviation': std_devs
        })
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Bull ratio plot
        sns.barplot(x='State', y='Bull Ratio', data=df, ax=ax1, palette='viridis')
        ax1.set_title('Bull Market Ratio by State', fontsize=16)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add horizontal line at 0.5 to indicate bull/bear threshold
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax1.text(len(states)-1, 0.52, 'Bull/Bear Threshold', color='red', ha='right')
        
        # Mean return and std deviation plot
        ax2.bar(states, mean_returns, yerr=std_devs, capsize=10, color='#1f77b4')
        ax2.set_title('Mean Return and Volatility by State', fontsize=16)
        ax2.set_ylabel('Mean Return')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(mean_returns):
            ax2.text(i, v + std_devs[i] + 0.5, f'{v:.2f}±{std_devs[i]:.2f}', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'state_interpretations.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"State interpretations plot saved to {output_dir / 'state_interpretations.png'}")
        
        return True
    except Exception as e:
        print(f"Error plotting state interpretations: {e}")
        return False

def plot_confusion_matrix():
    """Plot confusion matrix from model results"""
    try:
        # Load model results
        with open('structured_emission_model_results.json', 'r') as f:
            results = json.load(f)
        
        # Extract confusion matrix
        conf_matrix = results.get('confusion_matrix', [[0, 0], [0, 0]])
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Bear', 'Bull'],
                   yticklabels=['Bear', 'Bull'])
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to {output_dir / 'confusion_matrix.png'}")
        
        return True
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        return False

def visualize_model_parameters():
    """Visualize the parameters of the trained HMM model"""
    try:
        # Load the model
        model_path = 'optimized_hmm_classification_model.pt'
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract model parameters
        transition_matrix = model_data['T'].numpy()
        emission_matrix = model_data['E'].numpy()
        initial_distribution = model_data['T0'].numpy()
        
        # Plot the parameters
        plot_transition_matrix(transition_matrix)
        plot_emission_matrix(emission_matrix)
        plot_initial_distribution(initial_distribution)
        
        return True
    except Exception as e:
        print(f"Error visualizing model parameters: {e}")
        return False

def run_all_visualizations():
    """Run all visualization functions"""
    print("\n" + "="*70)
    print("CREATING MODEL VISUALIZATIONS")
    print("="*70)
    
    # Run all visualization functions
    visualize_model_parameters()
    plot_performance_comparison()
    plot_state_interpretations()
    plot_confusion_matrix()
    
    print("\nAll visualizations have been created and saved to the 'model_visualizations' directory.")
    print("You can include these visualizations in your reports and presentations.")

if __name__ == "__main__":
    run_all_visualizations()
