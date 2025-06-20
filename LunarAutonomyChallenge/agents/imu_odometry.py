import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

class InertialOdometry:
    def __init__(self, initial_position=None, initial_orientation=None):
        """
        Initialize inertial odometry

        Args: initial_position (list or np.array): [x, y, z] initial position (default [0, 0, 0]),
              initial_orientation (list or np.array): [roll, pitch, yaw] initial orientation (default [0, 0, 0]).
        """

        # Default Values
        if initial_position is None:
            initial_position = np.zeros(3)
        if initial_orientation is None:
            initial_orientation = no.zeros(3)

        # Lunar Gravity in m/s^2
        self.G = np.array([0, 0, 1.62])

        # Create transformation matrix from initial position and orientation
        self.rover2world = pt.transform_from(
                pr.matrix_from_euler(
                    [
                        initial_orientation[2],
                        initial_orientation[1],
                        initial_orientation[0],
                    ],
                    2,
                    1,
                    0,
                    extrinsic=False,

                ),

                # Initialize velocities for integration
                self.vx, self.vy, self.vz = 0.0, 0.0, 0.0

                # Previous timestamp for dt calculation
                self.prev_time = None


