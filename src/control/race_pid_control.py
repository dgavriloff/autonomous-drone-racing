"""PID control for RACE drone.

Based on DSLPIDControl from gym-pybullet-drones, adapted for the RACE drone model.
The RACE drone has different mass (830g vs 27g) and thrust characteristics.
"""

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class RacePIDControl(BaseControl):
    """PID control class for RACE drone.

    Adapted from DSLPIDControl with tuning for the larger, more powerful RACE drone.
    RACE drone specs: 830g mass, 200 km/h max speed, thrust2weight=4.17
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        """Initialize RACE PID controller.

        Parameters
        ----------
        drone_model : DroneModel
            Should be DroneModel.RACE
        g : float
            Gravitational acceleration
        """
        super().__init__(drone_model=drone_model, g=g)

        # RACE drone is ~31x heavier than CF2X (830g vs 27g)
        # But has higher thrust-to-weight ratio (4.17 vs 2.25), so it's more responsive
        # Start with conservative gains and tune from there

        # Position control gains (conservative for stability)
        # Original CF2X: P=[.4, .4, 1.25], I=[.05, .05, .05], D=[.2, .2, .5]
        # Scale by sqrt(mass_ratio) for position control (force ~ mass * accel)
        scale_p = 5.0  # Conservative position control
        self.P_COEFF_FOR = np.array([.4, .4, 1.25]) * scale_p
        self.I_COEFF_FOR = np.array([.05, .05, .05]) * scale_p
        self.D_COEFF_FOR = np.array([.2, .2, .5]) * scale_p * 2.0  # Extra damping

        # Attitude control gains (scale with inertia ratio)
        # Original CF2X: P=[70000, 70000, 60000], I=[0, 0, 500], D=[20000, 20000, 12000]
        # CF2X inertia: ~1.4e-5, RACE inertia: ~3.1e-3, ratio ~220
        # REDUCED: Previous 150x caused roll-pitch coupling instability
        # Using 20x with heavy D-damping to prevent oscillations
        inertia_ratio = 20  # Much more conservative
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.]) * inertia_ratio
        self.I_COEFF_TOR = np.array([.0, .0, 500.]) * inertia_ratio * 0.2  # Minimal integral
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.]) * inertia_ratio * 4.0  # Heavy damping

        # PWM to RPM conversion for RACE drone
        # For hover: thrust = mass * g = 0.830 * 9.8 = 8.134 N
        # Each motor thrust: kf * rpm^2, total = 4 * kf * rpm^2
        # Hover RPM: rpm = sqrt(mg / (4 * kf)) = sqrt(8.134 / (4 * 8.47e-9)) = 15,510
        #
        # Use same PWM scale as CF2X but adjust const to match hover point
        # At PWM=35000 (mid-range), we want rpm ~= 15500
        # rpm = scale * pwm + const -> 15500 = 0.2685 * 35000 + const
        # const = 15500 - 9397 = 6103
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 6100.0
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        # At MIN_PWM: rpm = 0.2685 * 20000 + 6100 = 11,470
        # At MAX_PWM: rpm = 0.2685 * 65535 + 6100 = 23,697

        # Mixer matrix for X configuration (same as CF2X)
        self.MIXER_MATRIX = np.array([
            [-.5, -.5, -1],
            [-.5,  .5,  1],
            [.5, .5, -1],
            [.5, -.5,  1]
        ])

        self.reset()

    def reset(self):
        """Reset controller state."""
        super().reset()
        self.last_rpy = np.zeros(3)
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """Compute PID control action as RPMs.

        Parameters
        ----------
        control_timestep : float
            Time step for control computation
        cur_pos : ndarray
            Current position (3,)
        cur_quat : ndarray
            Current orientation as quaternion (4,)
        cur_vel : ndarray
            Current velocity (3,)
        cur_ang_vel : ndarray
            Current angular velocity (3,)
        target_pos : ndarray
            Target position (3,)
        target_rpy : ndarray
            Target roll, pitch, yaw (3,)
        target_vel : ndarray
            Target velocity (3,)
        target_rpy_rates : ndarray
            Target angular rates (3,)

        Returns
        -------
        ndarray
            Motor RPMs (4,)
        ndarray
            Position error (3,)
        float
            Yaw error
        """
        self.control_counter += 1

        thrust, computed_target_rpy, pos_e = self._positionControl(
            control_timestep, cur_pos, cur_quat, cur_vel,
            target_pos, target_rpy, target_vel
        )

        rpm = self._attitudeControl(
            control_timestep, thrust, cur_quat,
            computed_target_rpy, target_rpy_rates
        )

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def _positionControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                         target_pos, target_rpy, target_vel):
        """PID position control."""
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel

        # Integral with anti-windup
        self.integral_pos_e = self.integral_pos_e + pos_e * control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)

        # PID thrust computation
        target_thrust = (
            np.multiply(self.P_COEFF_FOR, pos_e) +
            np.multiply(self.I_COEFF_FOR, self.integral_pos_e) +
            np.multiply(self.D_COEFF_FOR, vel_e) +
            np.array([0, 0, self.GRAVITY])
        )

        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:, 2]))
        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        # Compute target orientation from thrust direction
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).transpose()

        target_euler = Rotation.from_matrix(target_rotation).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print(f"[WARNING] Control iteration {self.control_counter}: euler angles outside [-pi, pi]")

        return thrust, target_euler, pos_e

    def _attitudeControl(self, control_timestep, thrust, cur_quat,
                         target_euler, target_rpy_rates):
        """PID attitude control."""
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        target_quat = Rotation.from_euler('XYZ', target_euler, degrees=False).as_quat()
        w, x, y, z = target_quat
        target_rotation = Rotation.from_quat([w, x, y, z]).as_matrix()

        # Rotation error
        rot_matrix_e = np.dot(target_rotation.transpose(), cur_rotation) - np.dot(cur_rotation.transpose(), target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])

        # Rate error
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy

        # Integral with anti-windup
        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        # PID torque computation
        target_torques = (
            -np.multiply(self.P_COEFF_TOR, rot_e) +
            np.multiply(self.D_COEFF_TOR, rpy_rates_e) +
            np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        )
        target_torques = np.clip(target_torques, -3200, 3200)

        # Mix thrust and torques to get motor PWMs
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
