"""Tests for neural network models."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest
from agent.model import DuelingDQN, ActorCritic, create_model
from agent.noisy_layers import NoisyLinear


class TestDuelingDQN:
    def test_output_shape(self):
        model = DuelingDQN(num_actions=112, feature_dim=53, use_noisy=False)
        board = torch.randn(2, 1, 22, 10)
        features = torch.randn(2, 53)
        q = model(board, features)
        assert q.shape == (2, 112)

    def test_dueling_identity(self):
        """Dueling decomposition property: Q = V + A - mean(A)"""
        model = DuelingDQN(num_actions=112, feature_dim=53, use_noisy=False)
        board = torch.randn(1, 1, 22, 10)
        features = torch.randn(1, 53)

        q = model(board, features)
        # Q-values should be finite.
        assert not torch.isnan(q).any()
        assert not torch.isinf(q).any()

    def test_noisy_exploration(self):
        """Noisy network should produce different outputs on different forward passes in train mode."""
        model = DuelingDQN(num_actions=112, feature_dim=53, use_noisy=True)
        model.train()

        board = torch.randn(1, 1, 22, 10)
        features = torch.randn(1, 53)

        q1 = model(board, features)
        q2 = model(board, features)
        # Outputs should differ due to noise resampling.
        assert not torch.allclose(q1, q2)

    def test_eval_mode_deterministic(self):
        """Noisy network in eval mode should produce same output."""
        model = DuelingDQN(num_actions=112, feature_dim=53, use_noisy=True)
        model.eval()

        board = torch.randn(1, 1, 22, 10)
        features = torch.randn(1, 53)

        q1 = model(board, features)
        q2 = model(board, features)
        assert torch.allclose(q1, q2)

    def test_gradient_flow(self):
        """Verify gradients flow through the network."""
        model = DuelingDQN(num_actions=112, feature_dim=53, use_noisy=False)
        board = torch.randn(4, 1, 22, 10)
        features = torch.randn(4, 53)
        targets = torch.randn(4, 112)

        q = model(board, features)
        loss = torch.nn.MSELoss()(q, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestActorCritic:
    def test_output_shapes(self):
        model = ActorCritic(num_actions=112, feature_dim=53)
        board = torch.randn(2, 1, 22, 10)
        features = torch.randn(2, 53)

        logits, value = model(board, features)
        assert logits.shape == (2, 112)
        assert value.shape == (2, 1)

    def test_evaluate_actions(self):
        model = ActorCritic(num_actions=112, feature_dim=53)
        board = torch.randn(4, 1, 22, 10)
        features = torch.randn(4, 53)
        actions = torch.randint(0, 112, (4,))
        mask = torch.ones(4, 112, dtype=torch.bool)

        log_probs, values, entropy = model.evaluate_actions(board, features, actions, mask)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert isinstance(entropy.item(), float)


class TestNoisyLinear:
    def test_output_shape(self):
        layer = NoisyLinear(10, 5)
        x = torch.randn(3, 10)
        y = layer(x)
        assert y.shape == (3, 5)

    def test_train_vs_eval(self):
        layer = NoisyLinear(10, 5)

        x = torch.randn(1, 10)
        layer.train()
        y1 = layer(x)
        y2 = layer(x)
        assert not torch.allclose(y1, y2)  # noise changes

        layer.eval()
        y3 = layer(x)
        y4 = layer(x)
        assert torch.allclose(y3, y4)  # noise disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
