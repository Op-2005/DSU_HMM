"""Unit tests for HMM model implementation."""

import pytest
import numpy as np
import torch
from src.models.hmm_model import HiddenMarkovModel


class TestHiddenMarkovModel:
    """Test cases for HiddenMarkovModel class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.num_states = 3
        self.num_observations = 5
        
        # Simple test parameters
        self.T = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5]
        ])
        self.E = np.ones((self.num_states, self.num_observations)) / self.num_observations
        self.T0 = np.array([0.6, 0.3, 0.1])
        
        self.hmm = HiddenMarkovModel(self.T, self.E, self.T0)
    
    def test_initialization(self):
        """Test HMM initialization."""
        assert self.hmm.S == self.num_states
        assert self.hmm.O == self.num_observations
        assert isinstance(self.hmm.T, torch.Tensor)
        assert isinstance(self.hmm.E, torch.Tensor)
        assert isinstance(self.hmm.T0, torch.Tensor)
    
    def test_viterbi_inference(self):
        """Test Viterbi algorithm."""
        observations = np.array([0, 1, 2, 1, 0])
        states, probs = self.hmm.viterbi_inference(observations)
        
        assert len(states) == len(observations)
        assert probs.shape == (len(observations), self.num_states)
        assert all(0 <= state < self.num_states for state in states)
    
    def test_forward_backward(self):
        """Test forward-backward algorithm."""
        observations = np.array([0, 1, 2, 1, 0])
        obs_prob_seq = self.hmm.E[:, observations].T  # Shape: (seq_len, num_states)
        
        self.hmm.forward_backward(obs_prob_seq)
        
        assert hasattr(self.hmm, 'forward')
        assert hasattr(self.hmm, 'backward')
        assert self.hmm.forward.shape == (len(observations), self.num_states)
        assert self.hmm.backward.shape == (len(observations), self.num_states)
    
    def test_invalid_input_shapes(self):
        """Test error handling for invalid input shapes."""
        with pytest.raises(Exception):
            # Invalid transition matrix shape
            T_invalid = np.ones((2, 3))
            HiddenMarkovModel(T_invalid, self.E, self.T0)
    
    @pytest.mark.slow
    def test_baum_welch_convergence(self):
        """Test Baum-Welch EM algorithm convergence."""
        observations = np.random.randint(0, self.num_observations, 100)
        
        T0_new, T_new, E_new, converged = self.hmm.Baum_Welch_EM(observations)
        
        assert isinstance(T0_new, torch.Tensor)
        assert isinstance(T_new, torch.Tensor)
        assert isinstance(E_new, torch.Tensor)
        assert isinstance(converged, bool)
        
        # Check probability constraints
        assert torch.allclose(T0_new.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(T_new.sum(dim=1), torch.ones(self.num_states), atol=1e-6)
        assert torch.allclose(E_new.sum(dim=1), torch.ones(self.num_states), atol=1e-6) 