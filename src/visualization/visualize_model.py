#!/usr/bin/env python
"""
Script to visualize HMM model parameters
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def visualize_model(model_path):
    """Load a PyTorch HMM model and visualize its parameters"""
    print(f"Loading model from {model_path}...")
    
    try:
        # Load the model
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Create a directory for the visualizations
        output_dir = Path('model_visualizations')
        output_dir.mkdir(exist_ok=True)
        
        # Set up the figure style
        plt.style.use('ggplot')
        
        # 1. Visualize the transition matrix
        if 'T' in model_data and isinstance(model_data['T'], torch.Tensor):
            T = model_data['T'].numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(T, annot=True, fmt='.2f', cmap='Blues', 
                        xticklabels=[f'State {i}' for i in range(T.shape[1])],
                        yticklabels=[f'State {i}' for i in range(T.shape[0])])
            plt.title('Transition Matrix (T)')
            plt.xlabel('To State')
            plt.ylabel('From State')
            plt.tight_layout()
            plt.savefig(output_dir / 'transition_matrix.png', dpi=300)
            plt.close()
            print(f"Saved transition matrix visualization to {output_dir / 'transition_matrix.png'}")
        
        # 2. Visualize the emission matrix
        if 'E' in model_data and isinstance(model_data['E'], torch.Tensor):
            E = model_data['E'].numpy()
            plt.figure(figsize=(15, 10))
            sns.heatmap(E, cmap='viridis', 
                        xticklabels=[f'Obs {i}' for i in range(E.shape[1])],
                        yticklabels=[f'State {i}' for i in range(E.shape[0])])
            plt.title('Emission Matrix (E)')
            plt.xlabel('Observation')
            plt.ylabel('State')
            plt.tight_layout()
            plt.savefig(output_dir / 'emission_matrix.png', dpi=300)
            plt.close()
            print(f"Saved emission matrix visualization to {output_dir / 'emission_matrix.png'}")
            
            # Also create line plots for each state's emission probabilities
            plt.figure(figsize=(15, 10))
            for i in range(E.shape[0]):
                plt.plot(range(E.shape[1]), E[i], marker='o', label=f'State {i}')
            plt.title('Emission Probabilities by State')
            plt.xlabel('Observation')
            plt.ylabel('Probability')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / 'emission_probabilities.png', dpi=300)
            plt.close()
            print(f"Saved emission probabilities visualization to {output_dir / 'emission_probabilities.png'}")
        
        # 3. Visualize the initial state distribution
        if 'T0' in model_data and isinstance(model_data['T0'], torch.Tensor):
            T0 = model_data['T0'].numpy()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=np.arange(len(T0)), y=T0)
            plt.title('Initial State Distribution (T0)')
            plt.xlabel('State')
            plt.ylabel('Probability')
            plt.xticks(np.arange(len(T0)), [f'State {i}' for i in range(len(T0))])
            plt.tight_layout()
            plt.savefig(output_dir / 'initial_state_distribution.png', dpi=300)
            plt.close()
            print(f"Saved initial state distribution visualization to {output_dir / 'initial_state_distribution.png'}")
            
        print(f"\nAll visualizations saved to {output_dir}")
            
    except Exception as e:
        print(f"Error visualizing model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PyTorch HMM model parameters")
    parser.add_argument("model_path", help="Path to the .pt model file")
    args = parser.parse_args()
    
    visualize_model(args.model_path)
