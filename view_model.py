#!/usr/bin/env python
"""
Script to view the contents of PyTorch .pt model files
"""

import torch
import argparse
import json
from pathlib import Path

def load_and_display_model(model_path):
    """Load a PyTorch model and display its contents"""
    print(f"Loading model from {model_path}...")
    
    try:
        # Load the model
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Print model type
        print(f"\nModel type: {type(model_data)}")
        
        # If it's a dictionary, show the keys
        if isinstance(model_data, dict):
            print("\nModel contains the following keys:")
            for key in model_data.keys():
                print(f"  - {key}")
            
            # For each key, show the type and shape if applicable
            print("\nDetailed contents:")
            for key, value in model_data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {type(value)} with shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
                
                # If it's a tensor, show some statistics
                if isinstance(value, torch.Tensor):
                    print(f"    Min: {value.min().item():.4f}, Max: {value.max().item():.4f}, Mean: {value.mean().item():.4f}")
                    
                    # Show a sample of the tensor (first few elements)
                    if value.numel() > 0:
                        flat_value = value.flatten()
                        sample_size = min(5, flat_value.numel())
                        print(f"    Sample values: {flat_value[:sample_size].tolist()}")
        
        # If it's a module, show its structure
        elif hasattr(model_data, '_modules'):
            print("\nModel structure:")
            print(model_data)
            
            # Try to get state dict
            try:
                state_dict = model_data.state_dict()
                print("\nModel state dictionary contains:")
                for key, value in state_dict.items():
                    print(f"  {key}: {type(value)} with shape {value.shape}")
            except:
                print("\nCouldn't access state dictionary")
        
        # Save a JSON representation of the model structure
        output_file = Path(model_path).with_suffix('.json')
        try:
            # Convert model data to a serializable format
            serializable_data = {}
            if isinstance(model_data, dict):
                for key, value in model_data.items():
                    if isinstance(value, torch.Tensor):
                        serializable_data[key] = {
                            'type': 'Tensor',
                            'shape': list(value.shape),
                            'min': float(value.min().item()),
                            'max': float(value.max().item()),
                            'mean': float(value.mean().item()),
                            'sample': value.flatten()[:5].tolist() if value.numel() > 0 else []
                        }
                    else:
                        serializable_data[key] = str(type(value))
            
                # Save to JSON
                with open(output_file, 'w') as f:
                    json.dump(serializable_data, f, indent=2)
                print(f"\nSaved model structure to {output_file}")
        except Exception as e:
            print(f"\nFailed to save JSON representation: {e}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View PyTorch model file contents")
    parser.add_argument("model_path", help="Path to the .pt model file")
    args = parser.parse_args()
    
    load_and_display_model(args.model_path)
