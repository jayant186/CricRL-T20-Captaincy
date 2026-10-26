""" Unit Tests for Deep Q-Network Agent"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dqn_agent import DQNAgent, QNetwork, ReplayBuffer

# 1. QNetwork Tests
class TestQNetwork:
    def test_forward_pass_batch(self):
        net = QNetwork(state_dim=6, action_dim=3, hidden_dims=(64, 64))
        batch = torch.randn(10, 6)
        out = net(batch)
        assert out.shape == (10, 3)
        assert torch.isfinite(out).all()

    def test_forward_pass_single(self):
        net = QNetwork(state_dim=6, action_dim=3)
        state = torch.randn(1, 6)
        out = net(state)
        assert out.shape == (1, 3)

# 2. ReplayBuffer Tests
class TestReplayBuffer:
    def test_buffer_push_and_length(self):
        buf = ReplayBuffer(capacity=100, state_dim=6)
        assert len(buf) == 0

        obs = np.ones(6, dtype=np.float32)
        buf.push(obs, 1, 0.5, obs * 2, False)
        assert len(buf) == 1

        # Fill past capacity to test circular wrapping
        for i in range(150):
            buf.push(obs * i, i % 3, float(i), obs * (i + 1), i % 10 == 0)

        assert len(buf) == 100

    def test_buffer_sample_tensors(self):
        buf = ReplayBuffer(capacity=50, state_dim=6)
        obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

        for i in range(30):
            buf.push(obs + i, i % 3, 1.0, obs + i + 1, False)

        device = torch.device("cpu")
        b_obs, b_act, b_rew, b_next_obs, b_dones = buf.sample(16, device)

        assert b_obs.shape == (16, 6)
        assert b_act.shape == (16, 1)
        assert b_rew.shape == (16, 1)
        assert b_next_obs.shape == (16, 6)
        assert b_dones.shape == (16, 1)
        assert b_obs.dtype == torch.float32
        assert b_act.dtype == torch.int64

# 3. DQNAgent Tests
class TestDQNAgent:
    def test_agent_initialization(self):
        agent = DQNAgent(state_dim=6, action_dim=3)
        assert agent.epsilon == 1.0
        assert agent.action_dim == 3
        assert agent.state_dim == 6

    def test_select_action_greedy_in_eval_mode(self):
        agent = DQNAgent(state_dim=6, action_dim=3, epsilon_start=1.0)
        obs = np.zeros(6, dtype=np.float32)

        # Force specific output weights to make action 2 clearly highest
        with torch.no_grad():
            for param in agent.q_network.parameters():
                param.zero_()
            # Set bias of last layer so action 2 is maximum
            agent.q_network.network[-1].bias.copy_(torch.tensor([0.0, 1.0, 5.0]))

        action = agent.select_action(obs, eval_mode=True)
        assert action == 2
        # Epsilon should not decay in eval mode
        assert agent.epsilon == 1.0

    def test_epsilon_decay_during_training(self):
        agent = DQNAgent(
            state_dim=6,
            action_dim=3,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=100,
        )
        obs = np.zeros(6, dtype=np.float32)

        for _ in range(50):
            agent.select_action(obs, eval_mode=False)

        assert 0.5 < agent.epsilon < 0.6

        for _ in range(60):
            agent.select_action(obs, eval_mode=False)

        assert agent.epsilon == pytest.approx(0.1, abs=1e-4)

    def test_agent_update_step(self):
        agent = DQNAgent(state_dim=6, action_dim=3, batch_size=16)
        obs = np.random.uniform(0.0, 1.0, size=6).astype(np.float32)

        # Before buffer has batch_size, update() should return None
        assert agent.update() is None

        # Fill buffer with synthetic transitions
        for i in range(40):
            agent.store_transition(
                obs,
                action=i % 3,
                reward=float(i % 2),
                next_obs=obs,
                done=(i % 10 == 0),
            )

        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0
        assert np.isfinite(loss)

    def test_save_and_load_checkpoint(self):
        agent1 = DQNAgent(state_dim=6, action_dim=3, lr=1e-3)
        obs = np.array([0.1, 0.5, 0.8, 0.4, 0.3, 0.2], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.pt"

            # Mutate weights
            for p in agent1.q_network.parameters():
                p.data.add_(0.5)

            agent1.save(ckpt_path)
            assert ckpt_path.exists()

            agent2 = DQNAgent(state_dim=6, action_dim=3)
            agent2.load(ckpt_path)

            # Compare outputs
            q1 = agent1.q_network(torch.as_tensor(obs).unsqueeze(0))
            q2 = agent2.q_network(torch.as_tensor(obs).unsqueeze(0))
            assert torch.allclose(q1, q2)

    def test_get_q_values(self):
        agent = DQNAgent(state_dim=6, action_dim=3)
        obs = np.array([0.1, 0.5, 0.8, 0.4, 0.3, 0.2], dtype=np.float32)
        q_vals = agent.get_q_values(obs)
        assert isinstance(q_vals, np.ndarray)
        assert q_vals.shape == (3,)
        assert np.isfinite(q_vals).all()
        # Verify greedy action matches argmax of q_values
        assert agent.select_action(obs, eval_mode=True) == int(np.argmax(q_vals))

    def test_weights_genuinely_update_via_backprop(self):
        """
        Verify that calling update() genuinely modifies network parameters
        through gradient descent and the Bellman loss.
        """
        agent = DQNAgent(state_dim=6, action_dim=3, lr=1e-2, batch_size=16)
        obs = np.random.uniform(0.1, 0.9, size=6).astype(np.float32)

        # Store transitions with positive rewards to ensure non-zero gradient
        for i in range(32):
            agent.store_transition(
                obs=obs + 0.01 * i,
                action=i % 3,
                reward=1.0 if (i % 2 == 0) else -1.0,
                next_obs=obs + 0.02 * i,
                done=False,
            )

        # Capture parameter weights before gradient update
        initial_params = [p.clone().detach() for p in agent.q_network.parameters()]

        loss = agent.update()
        assert loss is not None and loss > 0.0

        # Verify at least one parameter tensor has changed significantly
        weights_changed = False
        for p_before, p_after in zip(initial_params, agent.q_network.parameters()):
            if not torch.allclose(p_before, p_after.detach(), atol=1e-5):
                weights_changed = True
                break
        assert weights_changed, "Network weights did not change after update()!"