"""
G&CNet - Guidance and Control Network.

Neural network controller that outputs direct motor RPM commands
from estimated state and gate information. Supports both imitation
learning from PID expert and reinforcement learning fine-tuning.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.layers(x))


class GCNet(nn.Module):
    """
    Guidance and Control Network.

    Input features:
    - Position (3): drone position in world frame
    - Velocity (3): drone velocity in world frame
    - Orientation (4): quaternion (w, x, y, z)
    - Angular velocity (3): body angular rates
    - Gate position (3): next gate position in world frame
    - Gate direction (3): unit vector through gate
    - Velocity target (3): desired velocity magnitude and direction
    Total: 22 features

    Output:
    - 4 motor RPMs (normalized to [0, 1])

    Architecture:
    - Input projection
    - 2 residual blocks (256 dims)
    - Output projection with sigmoid activation
    """

    # Input feature indices
    POS_IDX = slice(0, 3)
    VEL_IDX = slice(3, 6)
    ORI_IDX = slice(6, 10)
    ANG_VEL_IDX = slice(10, 13)
    GATE_POS_IDX = slice(13, 16)
    GATE_DIR_IDX = slice(16, 19)
    VEL_TARGET_IDX = slice(19, 22)

    INPUT_DIM = 22
    OUTPUT_DIM = 4

    def __init__(
        self,
        hidden_dims: List[int] = [256, 256],
        use_residual: bool = True,
        dropout: float = 0.0,
        max_rpm: float = 21702.64,  # CF2X MAX_RPM from gym-pybullet-drones
    ):
        """
        Initialize G&CNet.

        Args:
            hidden_dims: Sizes of hidden layers
            use_residual: Whether to use residual connections
            dropout: Dropout rate
            max_rpm: Maximum motor RPM for denormalization
        """
        super().__init__()

        self.hidden_dims = hidden_dims
        self.use_residual = use_residual
        self.max_rpm = max_rpm

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(inplace=True),
        )

        # Hidden layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ])

        if use_residual and len(hidden_dims) >= 2:
            # Add one more residual block
            layers.append(ResidualBlock(hidden_dims[-1], dropout))

        self.hidden = nn.Sequential(*layers)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], self.OUTPUT_DIM),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, INPUT_DIM)

        Returns:
            Normalized motor RPMs (B, 4) in [0, 1]
        """
        x = self.input_proj(x)
        x = self.hidden(x)
        return self.output(x)

    def predict_rpms(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict actual motor RPMs.

        Args:
            x: Input tensor (B, INPUT_DIM)

        Returns:
            Motor RPMs (B, 4) in [0, max_rpm]
        """
        normalized = self.forward(x)
        return normalized * self.max_rpm

    @torch.no_grad()
    def get_action(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        orientation: np.ndarray,
        angular_velocity: np.ndarray,
        gate_position: np.ndarray,
        gate_direction: np.ndarray,
        velocity_target: np.ndarray,
    ) -> np.ndarray:
        """
        Get motor RPM action from state.

        Args:
            position: Drone position [x, y, z]
            velocity: Drone velocity [vx, vy, vz]
            orientation: Quaternion (w, x, y, z)
            angular_velocity: Body angular rates [wx, wy, wz]
            gate_position: Next gate position [x, y, z]
            gate_direction: Unit vector through gate
            velocity_target: Desired velocity [vx, vy, vz]

        Returns:
            Motor RPMs [rpm0, rpm1, rpm2, rpm3]
        """
        # Assemble input
        features = np.concatenate([
            position,
            velocity,
            orientation,
            angular_velocity,
            gate_position,
            gate_direction,
            velocity_target,
        ])

        # Convert to tensor and move to same device as model
        x = torch.from_numpy(features).float().unsqueeze(0)
        x = x.to(next(self.parameters()).device)

        # Predict
        rpms = self.predict_rpms(x)

        return rpms.cpu().numpy().flatten()

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GCNetDataset(Dataset):
    """Dataset for imitation learning."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        max_rpm: float = 21702.64,  # CF2X MAX_RPM from gym-pybullet-drones
    ):
        """
        Initialize dataset.

        Args:
            states: (N, 22) state features
            actions: (N, 4) motor RPM actions
            max_rpm: Maximum RPM for normalization
        """
        self.states = states.astype(np.float32)
        self.actions = (actions / max_rpm).astype(np.float32)  # Normalize to [0, 1]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.states[idx]),
            torch.from_numpy(self.actions[idx]),
        )


class GCNetTrainer:
    """Training manager for G&CNet imitation learning."""

    def __init__(
        self,
        model: GCNet,
        device: str = "auto",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize trainer.

        Args:
            model: GCNet model to train
            device: Device to use
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
        """
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for states, actions in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(states)
            loss = self.criterion(pred, actions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)

                pred = self.model(states)
                loss = self.criterion(pred, actions)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress

        Returns:
            Training history
        """
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_loss = 0.0
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}"
                if val_loader is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                print(msg)

            # Save best model
            if save_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(save_dir, "best_model.pt")

        if save_dir is not None:
            self.save(save_dir, "final_model.pt")

        return self.history

    def save(self, save_dir: str, filename: str = "model.pt"):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": {
                "hidden_dims": self.model.hidden_dims,
                "use_residual": self.model.use_residual,
                "max_rpm": self.model.max_rpm,
            },
        }

        torch.save(checkpoint, save_dir / filename)

    def load(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)


class GCNetRLWrapper(nn.Module):
    """
    Wrapper for using GCNet with stable-baselines3.

    Provides continuous action space output with proper scaling
    for RL training.
    """

    def __init__(
        self,
        observation_dim: int = 22,
        action_dim: int = 4,
        hidden_dims: List[int] = [256, 256],
        log_std_init: float = -1.0,
    ):
        """
        Initialize RL wrapper.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer sizes
            log_std_init: Initial log standard deviation
        """
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        # Shared feature extractor
        layers = [
            nn.Linear(observation_dim, hidden_dims[0]),
            nn.ReLU(),
        ]
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.ReLU(),
            ])

        self.features = nn.Sequential(*layers)

        # Policy head (mean)
        self.policy_mean = nn.Linear(hidden_dims[-1], action_dim)

        # Log std (learnable parameter)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Value head
        self.value = nn.Linear(hidden_dims[-1], 1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Observation tensor

        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        features = self.features(x)
        action_mean = torch.sigmoid(self.policy_mean(features))  # [0, 1]
        value = self.value(features)
        return action_mean, self.log_std.expand_as(action_mean), value

    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            x: Observation tensor
            deterministic: If True, return mean action

        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std, _ = self(x)

        if deterministic:
            return mean, torch.zeros_like(mean[:, 0])

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()
        action = torch.clamp(action, 0, 1)

        log_prob = normal.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions.

        Args:
            x: Observation tensor
            actions: Actions to evaluate

        Returns:
            Tuple of (log_prob, entropy, value)
        """
        mean, log_std, value = self(x)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(actions).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)

        return log_prob, entropy, value.squeeze(-1)


def collect_expert_data(
    num_steps: int = 100000,
    env_name: str = "racing",
    ctrl_freq: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect expert demonstrations using PID controller.

    Args:
        num_steps: Number of steps to collect
        env_name: Environment name
        ctrl_freq: Control frequency in Hz (should match inference frequency)

    Returns:
        Tuple of (states, actions) arrays
    """
    from gym_pybullet_drones.envs import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

    # Create environment with specified control frequency
    # CRITICAL: Must match inference frequency (500Hz for HighFreqRacingAviary)
    pyb_freq = max(1000, ctrl_freq * 4)  # PyBullet needs 4x control freq minimum
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        physics=Physics.PYB,
        gui=False,
        ctrl_freq=ctrl_freq,
        pyb_freq=pyb_freq,
    )

    print(f"Expert data collection: ctrl_freq={ctrl_freq}Hz, pyb_freq={pyb_freq}Hz")

    # Create controller
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    # Storage
    states_list = []
    actions_list = []

    # Define simple gate sequence
    gates = [
        np.array([2, 0, 1]),
        np.array([4, 2, 1.2]),
        np.array([4, 4, 1]),
        np.array([2, 4, 0.8]),
        np.array([0, 2, 1]),
    ]

    obs, info = env.reset()
    gate_idx = 0
    steps = 0

    print(f"Collecting {num_steps} expert demonstrations...")

    while steps < num_steps:
        # Get state
        state = env._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        quat = state[3:7]  # (x, y, z, w) in pybullet
        ang_vel = state[13:16]

        # Convert quaternion to (w, x, y, z)
        quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

        # Gate info
        gate_pos = gates[gate_idx]
        gate_dir = (gate_pos - pos) / (np.linalg.norm(gate_pos - pos) + 1e-6)

        # Velocity target (towards gate)
        target_speed = 3.0
        vel_target = gate_dir * target_speed

        # Assemble state features
        features = np.concatenate([
            pos, vel, quat_wxyz, ang_vel,
            gate_pos, gate_dir, vel_target,
        ])

        # Compute PID action
        target_rpy = np.zeros(3)
        action, _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state,
            target_pos=gate_pos,
            target_rpy=target_rpy,
        )

        states_list.append(features)
        actions_list.append(action.flatten())

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action.reshape(1, 4))
        steps += 1

        # Check gate progress
        if np.linalg.norm(pos - gate_pos) < 0.5:
            gate_idx = (gate_idx + 1) % len(gates)

        # Reset if crashed or out of bounds
        crashed = pos[2] < 0.05 or pos[2] > 3.0 or np.abs(pos[0]) > 10 or np.abs(pos[1]) > 10
        if terminated or truncated or crashed:
            obs, info = env.reset()
            # Randomize which gate to target for diversity
            gate_idx = np.random.randint(0, len(gates))
            # Skip a few steps to let drone stabilize with hover thrust
            hover_rpm = 14500  # ~hover
            hover_action = np.array([[hover_rpm, hover_rpm, hover_rpm, hover_rpm]])
            for _ in range(50):  # More steps to stabilize
                obs, _, _, _, _ = env.step(hover_action)

        if steps % 10000 == 0:
            print(f"  Collected {steps}/{num_steps} steps")

    env.close()

    states = np.array(states_list)
    actions = np.array(actions_list)

    print(f"Collection complete: {states.shape[0]} samples")
    return states, actions


def create_gcnet(
    pretrained_path: Optional[str] = None,
    device: str = "auto",
) -> GCNet:
    """
    Create GCNet model, optionally loading pretrained weights.

    Args:
        pretrained_path: Path to pretrained checkpoint
        device: Device to use

    Returns:
        GCNet model
    """
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        # Use config from checkpoint if available
        config = checkpoint.get("config", {})
        model = GCNet(
            hidden_dims=config.get("hidden_dims", [256, 256]),
            use_residual=config.get("use_residual", True),
            max_rpm=config.get("max_rpm", 21702.64),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = GCNet()

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test G&CNet
    print("Testing G&CNet architecture...")

    model = GCNet()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    batch = torch.randn(2, GCNet.INPUT_DIM)
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Test get_action
    rpms = model.get_action(
        position=np.array([0, 0, 1]),
        velocity=np.array([1, 0, 0]),
        orientation=np.array([1, 0, 0, 0]),
        angular_velocity=np.array([0, 0, 0]),
        gate_position=np.array([5, 0, 1]),
        gate_direction=np.array([1, 0, 0]),
        velocity_target=np.array([3, 0, 0]),
    )
    print(f"\nPredicted RPMs: {rpms}")

    # Test training with random data
    print("\nTesting imitation learning pipeline...")

    num_samples = 1000
    states = np.random.randn(num_samples, GCNet.INPUT_DIM).astype(np.float32)
    actions = np.random.uniform(0, 65535, (num_samples, 4)).astype(np.float32)

    dataset = GCNetDataset(states, actions)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    trainer = GCNetTrainer(model, device="cpu")
    history = trainer.train(loader, epochs=5, verbose=True)

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print("\nTest complete!")
