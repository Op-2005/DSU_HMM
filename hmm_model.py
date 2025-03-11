import torch
import numpy as np
import time


class HiddenMarkovModel(object):
    def __init__(self, T, E, T0, device='cpu', epsilon=0.001, maxStep=10):
        self.device = 'cpu'
        self.maxStep = maxStep
        self.epsilon = epsilon
        self.S = T.shape[0]
        self.O = E.shape[0]
        self.prob_state_1 = []
        
        # Convert to NumPy arrays first if they're not already
        if isinstance(T, torch.Tensor):
            T = T.detach().cpu().numpy()
        if isinstance(E, torch.Tensor):
            E = E.detach().cpu().numpy()
        if isinstance(T0, torch.Tensor):
            T0 = T0.detach().cpu().numpy()
            
        # Create tensors with better handling
        try:
            self.E = torch.tensor(E, dtype=torch.float64)
            self.T = torch.tensor(T, dtype=torch.float64)
            self.T0 = torch.tensor(T0, dtype=torch.float64)
        except Exception as e:
            print(f"Error initializing HMM tensors: {str(e)}")
            print(f"T shape: {T.shape}, E shape: {E.shape}, T0 shape: {T0.shape}")
            raise

    def initialize_viterbi_variables(self, shape):
        try:
            pathStates = torch.zeros(shape, dtype=torch.float64)
            pathScores = torch.zeros_like(pathStates)
            states_seq = torch.zeros([shape[0]], dtype=torch.int64)
            return pathStates, pathScores, states_seq
        except Exception as e:
            print(f"Error initializing Viterbi variables: {str(e)}")
            print(f"Shape: {shape}")
            raise

    def belief_propagation(self, scores):
        try:
            return scores.view(-1, 1) + torch.log(self.T)
        except Exception as e:
            print(f"Error in belief propagation: {str(e)}")
            print(f"Scores shape: {scores.shape}, T shape: {self.T.shape}")
            raise

    def viterbi_inference(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)

        self.N = len(x)
        shape = [self.N, self.S]
        
        try:
            pathStates, pathScores, states_seq = self.initialize_viterbi_variables(
                shape)
            obs_prob_full = torch.log(self.E[x])
            pathScores[0] = torch.log(self.T0) + obs_prob_full[0]

            for step, obs_prob in enumerate(obs_prob_full[1:]):
                belief = self.belief_propagation(pathScores[step, :])
                pathStates[step + 1] = torch.argmax(belief, 0)
                pathScores[step + 1] = torch.max(belief, 0)[0] + obs_prob

            states_seq[self.N - 1] = torch.argmax(pathScores[self.N-1, :], 0)

            for step in range(self.N - 1, 0, -1):
                state = states_seq[step]
                state_prob = pathStates[step][state]
                states_seq[step - 1] = state_prob

            return states_seq, torch.exp(pathScores)
        except Exception as e:
            print(f"Error in Viterbi inference: {str(e)}")
            print(f"x shape: {x.shape}, x unique values: {torch.unique(x)}")
            raise

    def initialize_forw_back_variables(self, shape):
        try:
            self.forward = torch.zeros(shape, dtype=torch.float64)
            self.backward = torch.zeros_like(self.forward)
            self.posterior = torch.zeros_like(self.forward)
        except Exception as e:
            print(f"Error initializing forward-backward variables: {str(e)}")
            print(f"Shape: {shape}")
            raise

    def _forward(self, obs_prob_seq):
        try:
            self.scale = torch.zeros([self.N], dtype=torch.float64)
            init_prob = self.T0 * obs_prob_seq[0]
            
            # Handle numerical issues
            sum_init = init_prob.sum()
            if sum_init > 0:
                self.scale[0] = 1.0 / sum_init
            else:
                print("Warning: Zero probability in forward algorithm initialization")
                self.scale[0] = 1.0
                
            self.forward[0] = self.scale[0] * init_prob

            for step, obs_prob in enumerate(obs_prob_seq[1:]):
                prev_prob = self.forward[step].unsqueeze(0)
                prior_prob = torch.matmul(prev_prob, self.T)
                forward_score = prior_prob * obs_prob
                forward_prob = torch.squeeze(forward_score)
                
                # Handle numerical issues
                sum_forward = forward_prob.sum()
                if sum_forward > 0:
                    self.scale[step + 1] = 1 / sum_forward
                else:
                    print(f"Warning: Zero probability in forward algorithm at step {step+1}")
                    self.scale[step + 1] = 1.0
                    
                self.forward[step + 1] = self.scale[step + 1] * forward_prob
        except Exception as e:
            print(f"Error in forward algorithm: {str(e)}")
            print(f"obs_prob_seq shape: {obs_prob_seq.shape}")
            raise

    def _backward(self, obs_prob_seq_rev):
        try:
            self.backward[0] = self.scale[self.N - 1] * \
                torch.ones([self.S], dtype=torch.float64)

            for step, obs_prob in enumerate(obs_prob_seq_rev[:-1]):
                next_prob = self.backward[step, :].unsqueeze(1)
                obs_prob_d = torch.diag(obs_prob)
                prior_prob = torch.matmul(self.T, obs_prob_d)
                backward_prob = torch.matmul(prior_prob, next_prob).squeeze()
                self.backward[step + 1] = self.scale[self.N -
                                                    2 - step] * backward_prob

            self.backward = torch.flip(self.backward, [0, 1])
        except Exception as e:
            print(f"Error in backward algorithm: {str(e)}")
            print(f"obs_prob_seq_rev shape: {obs_prob_seq_rev.shape}")
            raise

    def forward_backward(self, obs_prob_seq):
        try:
            self._forward(obs_prob_seq)
            obs_prob_seq_rev = torch.flip(obs_prob_seq, [0, 1])
            self._backward(obs_prob_seq_rev)
        except Exception as e:
            print(f"Error in forward-backward algorithm: {str(e)}")
            raise

    def re_estimate_transition(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)

        self.M = torch.zeros([self.N - 1, self.S, self.S], dtype=torch.float64)

        for t in range(self.N - 1):
            tmp_0 = torch.matmul(self.forward[t].unsqueeze(0), self.T)
            tmp_1 = tmp_0 * self.E[x[t + 1]].unsqueeze(0)
            denom = torch.matmul(
                tmp_1, self.backward[t + 1].unsqueeze(1)).squeeze()

            trans_re_estimate = torch.zeros(
                [self.S, self.S], dtype=torch.float64)

            for i in range(self.S):
                numer = self.forward[t, i] * self.T[i, :] * \
                    self.E[x[t+1]] * self.backward[t+1]
                trans_re_estimate[i] = numer / denom

            self.M[t] = trans_re_estimate

        self.gamma = self.M.sum(2).squeeze()
        T_new = self.M.sum(0) / self.gamma.sum(0).unsqueeze(1)
        T0_new = self.gamma[0, :]
        prod = (self.forward[self.N-1] * self.backward[self.N-1]).unsqueeze(0)
        s = prod / prod.sum()
        self.gamma = torch.cat([self.gamma, s], 0)
        self.prob_state_1.append(self.gamma[:, 0].detach().numpy())

        return T0_new, T_new

    def re_estimate_emission(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.int64)

        states_marginal = self.gamma.sum(0)
        seq_one_hot = torch.zeros([len(x), self.O], dtype=torch.float64)
        seq_one_hot.scatter_(1, x.unsqueeze(1), 1)
        emission_score = torch.matmul(seq_one_hot.transpose(1, 0), self.gamma)

        return emission_score / states_marginal

    def check_convergence(self, new_T0, new_transition, new_emission):
        with torch.no_grad():
            delta_T0 = torch.max(torch.abs(self.T0 - new_T0)
                                 ).item() < self.epsilon
            delta_T = torch.max(
                torch.abs(self.T - new_transition)).item() < self.epsilon
            delta_E = torch.max(
                torch.abs(self.E - new_emission)).item() < self.epsilon

        return delta_T0 and delta_T and delta_E

    def expectation_maximization_step(self, obs_seq):
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.int64)

        obs_prob_seq = self.E[obs_seq]
        self.forward_backward(obs_prob_seq)
        new_T0, new_transition = self.re_estimate_transition(obs_seq)
        new_emission = self.re_estimate_emission(obs_seq)
        converged = self.check_convergence(
            new_T0, new_transition, new_emission)

        self.T0 = new_T0
        self.E = new_emission
        self.T = new_transition

        return converged

    def Baum_Welch_EM(self, obs_seq):
        if not isinstance(obs_seq, torch.Tensor):
            obs_seq = torch.tensor(obs_seq, dtype=torch.int64)

        try:
            self.N = len(obs_seq)
            shape = [self.N, self.S]
            self.initialize_forw_back_variables(shape)
            converged = False

            start_time = time.time()
            print(f"Starting Baum-Welch EM with {self.maxStep} max steps")

            for i in range(self.maxStep):
                iter_start = time.time()
                try:
                    converged = self.expectation_maximization_step(obs_seq)
                    iter_time = time.time() - iter_start
                    print(f"  Step {i+1}/{self.maxStep} completed in {iter_time:.2f}s")

                    if converged:
                        print(f'Converged at step {i+1}')
                        break
                except Exception as step_error:
                    print(f"Error in EM step {i+1}: {str(step_error)}")
                    # Continue with next iteration instead of failing completely
                    continue

            total_time = time.time() - start_time
            print(f"Total training time: {total_time:.2f} seconds")

            return self.T0, self.T, self.E, converged
        except Exception as e:
            print(f"Error in Baum-Welch algorithm: {str(e)}")
            print(f"obs_seq shape: {obs_seq.shape}, unique values: {torch.unique(obs_seq)}")
            # Return current parameters even if we failed
            return self.T0, self.T, self.E, False

    def save_model(self, filepath):
        torch.save({
            'T': self.T,
            'E': self.E,
            'T0': self.T0,
            'S': self.S,
            'O': self.O,
            'prob_state_1': self.prob_state_1
        }, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        checkpoint = torch.load(filepath, map_location='cpu')
        T = checkpoint['T'].numpy()
        E = checkpoint['E'].numpy()
        T0 = checkpoint['T0'].numpy()

        model = cls(T, E, T0, device='cpu')
        model.prob_state_1 = checkpoint.get('prob_state_1', [])
        print(f"Model loaded from {filepath}")
        return model

    def interpret_states(self, states_seq, observations, actual_labels=None):
        unique_states = torch.unique(states_seq).numpy()
        state_interpretations = {}

        for state in unique_states:
            state_mask = (states_seq.numpy() == state)
            state_obs = observations[state_mask]
            mean_value = np.mean(state_obs)
            std_value = np.std(state_obs)

            if actual_labels is not None:
                state_labels = actual_labels[state_mask]
                bull_ratio = np.mean(state_labels)

                if bull_ratio > 0.7:
                    state_type = "Bull Market"
                elif bull_ratio < 0.3:
                    state_type = "Bear Market"
                else:
                    state_type = "Sideways/Mixed Market"

                state_interpretations[state] = {
                    'type': state_type,
                    'bull_ratio': bull_ratio,
                    'mean': mean_value,
                    'std': std_value
                }
            else:
                if mean_value > 0:
                    state_type = "Likely Bull Market"
                elif mean_value < 0:
                    state_type = "Likely Bear Market"
                else:
                    state_type = "Likely Sideways Market"

                state_interpretations[state] = {
                    'type': state_type,
                    'mean': mean_value,
                    'std': std_value
                }

        return state_interpretations

    def predict_one_step_ahead(self, current_state_probs, observation_map=None):
        next_state_probs = torch.matmul(
            current_state_probs.unsqueeze(0), self.T).squeeze(0)
        next_obs_probs = torch.matmul(
            next_state_probs.unsqueeze(0), self.E.T).squeeze(0)
        next_obs_probs_np = next_obs_probs.numpy()

        if observation_map is not None:
            prediction = np.sum(next_obs_probs_np * observation_map)
            return next_obs_probs_np, prediction
        else:
            return next_obs_probs_np, None

    def evaluate(self, observations, mode='classification', actual_values=None, actual_labels=None, observation_map=None, class_threshold=0.5, direct_states=False):
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.int64)

        states_seq, state_probs = self.viterbi_inference(observations)
        states_seq_np = states_seq.numpy()
        metrics = {'states_seq': states_seq_np}

        if mode == 'classification':
            if actual_labels is None:
                raise ValueError(
                    "actual_labels must be provided for classification mode")

            if direct_states:
                # Use states directly to predict labels
                # Calculate correlation between each state and bull/bear markets
                unique_states = np.unique(states_seq_np)
                state_correlations = {}

                for state in unique_states:
                    # Create a binary array for this state (1 where this state is active)
                    state_presence = (states_seq_np == state).astype(int)
                    # Calculate correlation with actual labels
                    corr = np.corrcoef(state_presence, actual_labels)[0, 1]
                    state_correlations[state] = corr

                # Assign states to bear/bull based on correlation
                state_to_label = {}
                for state in unique_states:
                    # If positive correlation with bull market (actual_labels==1), then predict bull
                    # If negative correlation, then predict bear
                    state_to_label[state] = 1 if state_correlations[state] >= 0 else 0

                pred_labels = np.array([state_to_label[state]
                                       for state in states_seq_np])

                # Print correlation information
                print("\nState to Market Regime Correlations:")
                for state, corr in state_correlations.items():
                    regime = "Bull" if state_to_label[state] == 1 else "Bear"
                    print(
                        f"  State {state}: {corr:.4f} correlation, classified as {regime} Market")
            else:
                # Original method - use majority voting within each state
                unique_states = np.unique(states_seq_np)
                state_to_label = {}

                for state in unique_states:
                    mask = (states_seq_np == state)
                    avg_label = np.mean(actual_labels[mask])
                    state_to_label[state] = 1 if avg_label >= class_threshold else 0

                pred_labels = np.array([state_to_label[state]
                                       for state in states_seq_np])

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

            accuracy = accuracy_score(actual_labels, pred_labels)
            precision = precision_score(
                actual_labels, pred_labels, zero_division=0)
            recall = recall_score(actual_labels, pred_labels, zero_division=0)
            f1 = f1_score(actual_labels, pred_labels, zero_division=0)
            conf_matrix = confusion_matrix(actual_labels, pred_labels)

            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'predicted_labels': pred_labels,
            })

            metrics['state_interpretations'] = self.interpret_states(
                states_seq, actual_values, actual_labels)

        elif mode == 'forecasting':
            if observation_map is None:
                raise ValueError(
                    "observation_map must be provided for forecasting mode")

            forecasts = []
            actual_next_values = actual_values[1:
                                               ] if actual_values is not None else None

            for t in range(len(observations) - 1):
                current_probs = torch.zeros(self.S, dtype=torch.float64)
                current_probs[states_seq[t]] = 1.0

                _, prediction = self.predict_one_step_ahead(
                    current_probs, observation_map)
                forecasts.append(prediction)

            forecasts = np.array(forecasts)

            if actual_next_values is not None:
                mse = np.mean((forecasts - actual_next_values) ** 2)
                mae = np.mean(np.abs(forecasts - actual_next_values))
                correlation = np.corrcoef(forecasts, actual_next_values)[0, 1]

                metrics.update({
                    'mse': mse,
                    'mae': mae,
                    'correlation': correlation,
                    'forecasts': forecasts,
                    'actual_next_values': actual_next_values
                })
            else:
                metrics['forecasts'] = forecasts

        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")

        return metrics
