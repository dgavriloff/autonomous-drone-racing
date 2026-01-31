"""
Motor Mixer for Crazyflie quadrotor.

Converts collective thrust and body torques to individual motor RPM commands
using the standard quadrotor mixing matrix.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class QuadConfig(Enum):
    """Quadrotor motor configuration."""
    X = "x"  # X configuration (standard Crazyflie)
    PLUS = "plus"  # + configuration


@dataclass
class DroneParams:
    """Physical parameters of the drone."""
    mass: float = 0.027  # kg (Crazyflie 2.X)
    arm_length: float = 0.0397  # meters
    kf: float = 3.16e-10  # Thrust coefficient [N/RPM^2]
    km: float = 7.94e-12  # Torque coefficient [Nm/RPM^2]
    ixx: float = 1.4e-5  # Moment of inertia x [kg*m^2]
    iyy: float = 1.4e-5  # Moment of inertia y [kg*m^2]
    izz: float = 2.17e-5  # Moment of inertia z [kg*m^2]
    max_rpm: float = 65535  # Maximum motor RPM
    min_rpm: float = 0  # Minimum motor RPM

    @property
    def gravity_thrust(self) -> float:
        """Thrust needed to hover [N]."""
        return self.mass * 9.81

    @property
    def hover_rpm(self) -> float:
        """RPM per motor for hover."""
        return np.sqrt(self.gravity_thrust / (4 * self.kf))


class MotorMixer:
    """
    Motor mixing for quadrotor control.

    Converts desired collective thrust and body torques into
    individual motor RPM commands using the standard mixing matrix.

    For X-configuration (Crazyflie):
    Motor numbering (top view):
        3   0
          X
        2   1

    Positive rotation: 0,2 CW, 1,3 CCW
    """

    def __init__(
        self,
        config: QuadConfig = QuadConfig.X,
        params: Optional[DroneParams] = None,
    ):
        """
        Initialize motor mixer.

        Args:
            config: Quadrotor configuration (X or +)
            params: Drone physical parameters
        """
        self.config = config
        self.params = params or DroneParams()

        # Compute mixing matrix
        self._compute_mixing_matrix()

        # Compute RPM limits
        self.min_rpm = self.params.min_rpm
        self.max_rpm = self.params.max_rpm

        # Precompute constants
        self.kf = self.params.kf
        self.km = self.params.km
        self.arm = self.params.arm_length

    def _compute_mixing_matrix(self):
        """
        Compute the mixing matrix for thrust/torque to motor forces.

        The mixing matrix M maps [thrust, tau_x, tau_y, tau_z] to [f0, f1, f2, f3]
        where f_i is the thrust from motor i.

        For X-config with arm length L and motors at 45 degrees:
        - Roll (tau_x): motors 0,1 vs 2,3
        - Pitch (tau_y): motors 0,3 vs 1,2
        - Yaw (tau_z): CW motors (0,2) vs CCW motors (1,3)
        """
        L = self.arm
        k_tau = self.km / self.kf  # Torque-to-thrust ratio

        if self.config == QuadConfig.X:
            # X configuration (45 degree offset)
            # Motor positions relative to CoM (normalized by arm length)
            # Motor 0: (+x, +y), Motor 1: (+x, -y)
            # Motor 2: (-x, -y), Motor 3: (-x, +y)
            d = L * np.sqrt(2) / 2  # Effective arm length for roll/pitch

            # Allocation matrix: [thrust, tau_x, tau_y, tau_z] -> [f0, f1, f2, f3]
            # Each row corresponds to one motor's contribution
            self.allocation_matrix = np.array([
                [1, -d, d, -k_tau],   # Motor 0 (CW)
                [1, -d, -d, k_tau],   # Motor 1 (CCW)
                [1, d, -d, -k_tau],   # Motor 2 (CW)
                [1, d, d, k_tau],     # Motor 3 (CCW)
            ]) / 4  # Divide by 4 since thrust is sum of all motors

        else:  # PLUS configuration
            self.allocation_matrix = np.array([
                [1, 0, L, -k_tau],    # Motor 0 (front, CW)
                [1, -L, 0, k_tau],    # Motor 1 (right, CCW)
                [1, 0, -L, -k_tau],   # Motor 2 (back, CW)
                [1, L, 0, k_tau],     # Motor 3 (left, CCW)
            ]) / 4

        # Compute inverse for going from motor forces to thrust/torques
        # (used for analysis, not control)
        self.inverse_allocation = np.linalg.pinv(self.allocation_matrix)

    def mix(
        self,
        thrust: float,
        torque: np.ndarray,
    ) -> np.ndarray:
        """
        Convert thrust and torques to motor RPMs.

        Args:
            thrust: Collective thrust [N]
            torque: Body torques [tau_x, tau_y, tau_z] [Nm]

        Returns:
            4-element array of motor RPMs [RPM]
        """
        # Form input vector
        u = np.array([thrust, torque[0], torque[1], torque[2]])

        # Compute motor forces
        motor_forces = self.allocation_matrix @ u

        # Convert forces to RPMs: F = kf * rpm^2 -> rpm = sqrt(F / kf)
        # Handle negative forces (clip to zero)
        motor_forces = np.maximum(motor_forces, 0)
        motor_rpms = np.sqrt(motor_forces / self.kf)

        # Clip to valid range
        motor_rpms = np.clip(motor_rpms, self.min_rpm, self.max_rpm)

        return motor_rpms

    def mix_normalized(
        self,
        thrust_normalized: float,
        torque_normalized: np.ndarray,
    ) -> np.ndarray:
        """
        Convert normalized thrust/torques to motor RPMs.

        Args:
            thrust_normalized: Thrust in [0, 1] range (0 = no thrust, 1 = max thrust)
            torque_normalized: Torques in [-1, 1] range

        Returns:
            4-element array of motor RPMs
        """
        # Convert normalized thrust to actual thrust
        max_thrust = 4 * self.kf * self.max_rpm**2
        thrust = thrust_normalized * max_thrust

        # Convert normalized torques to actual torques
        max_torque_roll = self.arm * self.kf * self.max_rpm**2
        max_torque_pitch = self.arm * self.kf * self.max_rpm**2
        max_torque_yaw = 2 * self.km * self.max_rpm**2

        torque = np.array([
            torque_normalized[0] * max_torque_roll,
            torque_normalized[1] * max_torque_pitch,
            torque_normalized[2] * max_torque_yaw,
        ])

        return self.mix(thrust, torque)

    def inverse_mix(self, motor_rpms: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert motor RPMs back to thrust and torques.

        Args:
            motor_rpms: 4-element array of motor RPMs

        Returns:
            Tuple of (thrust [N], torques [Nm])
        """
        # Convert RPMs to forces
        motor_forces = self.kf * motor_rpms**2

        # Compute thrust and torques
        u = self.inverse_allocation @ motor_forces

        thrust = u[0]
        torque = u[1:4]

        return thrust, torque

    def get_hover_rpm(self) -> float:
        """Get the RPM per motor needed to hover."""
        return self.params.hover_rpm

    def get_max_thrust(self) -> float:
        """Get maximum collective thrust [N]."""
        return 4 * self.kf * self.max_rpm**2

    def get_max_torque(self) -> np.ndarray:
        """Get maximum torques [Nm] for [roll, pitch, yaw]."""
        # Maximum torque when two motors at max, two at min
        max_torque_roll = self.arm * self.kf * self.max_rpm**2
        max_torque_pitch = self.arm * self.kf * self.max_rpm**2
        max_torque_yaw = 2 * self.km * self.max_rpm**2

        return np.array([max_torque_roll, max_torque_pitch, max_torque_yaw])


class AttitudeController:
    """
    Simple PD attitude controller for body rate control.

    Used to convert desired body rates to torque commands,
    which are then passed to the motor mixer.
    """

    def __init__(
        self,
        params: Optional[DroneParams] = None,
        kp_roll: float = 70000,
        kp_pitch: float = 70000,
        kp_yaw: float = 60000,
        kd_roll: float = 20000,
        kd_pitch: float = 20000,
        kd_yaw: float = 12000,
    ):
        """
        Initialize attitude controller.

        Args:
            params: Drone parameters
            kp_*: Proportional gains for roll/pitch/yaw
            kd_*: Derivative gains for roll/pitch/yaw
        """
        self.params = params or DroneParams()

        self.kp = np.array([kp_roll, kp_pitch, kp_yaw])
        self.kd = np.array([kd_roll, kd_pitch, kd_yaw])

        # Inertia matrix
        self.inertia = np.diag([
            self.params.ixx,
            self.params.iyy,
            self.params.izz,
        ])

    def compute_torque(
        self,
        current_euler: np.ndarray,
        current_rates: np.ndarray,
        target_euler: np.ndarray,
        target_rates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute torque command for attitude control.

        Args:
            current_euler: Current [roll, pitch, yaw] [rad]
            current_rates: Current angular rates [rad/s]
            target_euler: Target [roll, pitch, yaw] [rad]
            target_rates: Target angular rates (optional, default zeros)

        Returns:
            Torque command [tau_x, tau_y, tau_z] [Nm]
        """
        if target_rates is None:
            target_rates = np.zeros(3)

        # Attitude error
        angle_error = target_euler - current_euler

        # Wrap yaw error to [-pi, pi]
        angle_error[2] = np.arctan2(np.sin(angle_error[2]), np.cos(angle_error[2]))

        # Rate error
        rate_error = target_rates - current_rates

        # PD control
        torque = self.kp * angle_error + self.kd * rate_error

        return torque


class PositionController:
    """
    PID position controller for thrust and attitude targets.

    Converts position/velocity targets to thrust and attitude commands.
    """

    def __init__(
        self,
        params: Optional[DroneParams] = None,
        kp_xy: float = 0.4,
        kp_z: float = 1.25,
        ki_xy: float = 0.05,
        ki_z: float = 0.05,
        kd_xy: float = 0.2,
        kd_z: float = 0.5,
    ):
        """
        Initialize position controller.

        Args:
            params: Drone parameters
            kp_*, ki_*, kd_*: PID gains for xy and z axes
        """
        self.params = params or DroneParams()

        self.kp = np.array([kp_xy, kp_xy, kp_z])
        self.ki = np.array([ki_xy, ki_xy, ki_z])
        self.kd = np.array([kd_xy, kd_xy, kd_z])

        # Integral error accumulator
        self.integral_error = np.zeros(3)
        self.integral_limit = np.array([2.0, 2.0, 0.15])

        # Gravity
        self.gravity = 9.81

    def compute_thrust_and_attitude(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_euler: np.ndarray,
        target_pos: np.ndarray,
        target_vel: Optional[np.ndarray] = None,
        dt: float = 0.002,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute thrust and target attitude from position error.

        Args:
            current_pos: Current position [x, y, z] [m]
            current_vel: Current velocity [vx, vy, vz] [m/s]
            current_euler: Current [roll, pitch, yaw] [rad]
            target_pos: Target position [m]
            target_vel: Target velocity (optional, default zeros) [m/s]
            dt: Time step [s]

        Returns:
            Tuple of (thrust [N], target_euler [roll, pitch, yaw] [rad])
        """
        if target_vel is None:
            target_vel = np.zeros(3)

        # Position error
        pos_error = target_pos - current_pos

        # Velocity error
        vel_error = target_vel - current_vel

        # Update integral (with anti-windup)
        self.integral_error += pos_error * dt
        self.integral_error = np.clip(
            self.integral_error,
            -self.integral_limit,
            self.integral_limit,
        )

        # PID output (desired acceleration)
        accel_desired = (
            self.kp * pos_error +
            self.ki * self.integral_error +
            self.kd * vel_error
        )

        # Add gravity compensation
        accel_desired[2] += self.gravity

        # Compute thrust magnitude
        thrust = self.params.mass * np.linalg.norm(accel_desired)

        # Compute desired attitude
        # The thrust vector should point along accel_desired
        thrust_dir = accel_desired / (np.linalg.norm(accel_desired) + 1e-6)

        # Current yaw
        yaw = current_euler[2]

        # Compute target roll and pitch
        # Roll: rotation about body x-axis
        # Pitch: rotation about body y-axis
        target_roll = np.arcsin(
            thrust_dir[0] * np.sin(yaw) - thrust_dir[1] * np.cos(yaw)
        )
        target_pitch = np.arctan2(
            thrust_dir[0] * np.cos(yaw) + thrust_dir[1] * np.sin(yaw),
            thrust_dir[2]
        )

        # Limit angles
        max_angle = np.radians(30)
        target_roll = np.clip(target_roll, -max_angle, max_angle)
        target_pitch = np.clip(target_pitch, -max_angle, max_angle)

        target_euler = np.array([target_roll, target_pitch, yaw])

        return thrust, target_euler

    def reset(self):
        """Reset integral error."""
        self.integral_error = np.zeros(3)


class FullController:
    """
    Complete cascaded control: Position -> Attitude -> Motor RPMs.
    """

    def __init__(
        self,
        params: Optional[DroneParams] = None,
        config: QuadConfig = QuadConfig.X,
    ):
        """
        Initialize full controller.

        Args:
            params: Drone parameters
            config: Quadrotor configuration
        """
        self.params = params or DroneParams()

        self.position_ctrl = PositionController(self.params)
        self.attitude_ctrl = AttitudeController(self.params)
        self.mixer = MotorMixer(config, self.params)

    def compute_rpms(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_euler: np.ndarray,
        current_rates: np.ndarray,
        target_pos: np.ndarray,
        target_vel: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        dt: float = 0.002,
    ) -> np.ndarray:
        """
        Compute motor RPMs from position target.

        Args:
            current_pos: Current position [m]
            current_vel: Current velocity [m/s]
            current_euler: Current [roll, pitch, yaw] [rad]
            current_rates: Current angular rates [rad/s]
            target_pos: Target position [m]
            target_vel: Target velocity [m/s]
            target_yaw: Target yaw angle [rad]
            dt: Time step [s]

        Returns:
            4-element array of motor RPMs
        """
        # Position control -> thrust and target attitude
        thrust, target_euler = self.position_ctrl.compute_thrust_and_attitude(
            current_pos, current_vel, current_euler, target_pos, target_vel, dt
        )
        target_euler[2] = target_yaw  # Override yaw

        # Attitude control -> torques
        torque = self.attitude_ctrl.compute_torque(
            current_euler, current_rates, target_euler
        )

        # Mix to motor RPMs
        rpms = self.mixer.mix(thrust, torque)

        return rpms

    def reset(self):
        """Reset controller state."""
        self.position_ctrl.reset()


if __name__ == "__main__":
    # Test motor mixer
    print("Testing MotorMixer...")

    mixer = MotorMixer()

    print(f"Drone parameters:")
    print(f"  Mass: {mixer.params.mass} kg")
    print(f"  Arm length: {mixer.params.arm_length} m")
    print(f"  Hover RPM: {mixer.params.hover_rpm:.1f}")
    print(f"  Max thrust: {mixer.get_max_thrust():.3f} N")
    print(f"  Max torques: {mixer.get_max_torque()}")

    # Test hover
    hover_thrust = mixer.params.gravity_thrust
    hover_torque = np.zeros(3)
    hover_rpms = mixer.mix(hover_thrust, hover_torque)
    print(f"\nHover test:")
    print(f"  Thrust: {hover_thrust:.4f} N")
    print(f"  RPMs: {hover_rpms}")

    # Verify inverse
    thrust_back, torque_back = mixer.inverse_mix(hover_rpms)
    print(f"  Inverse thrust: {thrust_back:.4f} N")
    print(f"  Inverse torque: {torque_back}")

    # Test with torque
    print(f"\nRoll torque test:")
    roll_torque = np.array([0.001, 0, 0])
    roll_rpms = mixer.mix(hover_thrust, roll_torque)
    print(f"  Torque: {roll_torque}")
    print(f"  RPMs: {roll_rpms}")

    # Test full controller
    print(f"\nFull controller test:")
    ctrl = FullController()

    rpms = ctrl.compute_rpms(
        current_pos=np.array([0, 0, 0.5]),
        current_vel=np.array([0, 0, 0]),
        current_euler=np.array([0, 0, 0]),
        current_rates=np.array([0, 0, 0]),
        target_pos=np.array([1, 0, 1]),
    )
    print(f"  Target: [1, 0, 1]")
    print(f"  RPMs: {rpms}")

    print("\nTest complete!")
