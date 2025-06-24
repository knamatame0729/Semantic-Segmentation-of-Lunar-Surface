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
                initial_position
                )

                # Initialize velocities for integration
                self.vx, self.vy, self.vz = 0.0, 0.0, 0.0

                # Previous timestamp for dt calculation
                self.prev_time = None

                # Get initial position and orientation from transform
                position, orientation = self._get_odom_from_transformation3d(self.rover2world)

                print(
                    f"[ODO] Initialized inertial odometry at position ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                    f"with orientaion (roll: {np.degrees(orientation[0]):.1f}°, pitch: {np.degrees(orientation[1]):.1f}°, yaw: {np.degrees(orientation[2]):.1f}°)"
                    )

        def update(self, imu_data, linear_speed, angular_speed, current_time):
            """
            Update the inertial odometry using a single IMU reading with pytransform3d.

            Args:
                imu_data (np.array or list): Array containing IMU data in order:
                    - imu_data[0]: ax - Accelerometer X in m/s^2
                    - imu_data[1]: ay - Accelerometer Y in m/s^2
                    - imu_data[2]: az - Accelerometer Z in m/s^2
                    - imu_data[3]: wx - Gyroscope X in rad/s
                    - imu_data[4]: wy - Gyroscope Y in rad/s
                    - imu_data[5]: wz - Gyroscope Z in rad/s
                linear_speed (float): Current linear speed
                angluar_speed (float): Current angular speed
                current_time (float): Current time in seconds

            Returns:
                tuple: (position, orientation) where
                position is a numpy array [x, y, z]
                orientation is a numpy array [roll, pitch, yaw] in radians.
            """

            current_position, current_orientation = self._get_odom_from_transformation3d(
                self.rover2world
            )

            # If this is the first update, simply initialize the privious time.
            if self.prev_time is None:
                self.prev_time = current_time
                return current_position, current_orientation

            # Calculate the time difference (dt)
            dt = current_time - self.prec_time
            self.prev_time = current_time

            # Skip update if dt is too small
            if dt < 1e-6:
                return current_position, current_orientation

            # Extract IMU data
            ax, ay, az = imu_data[0:3]
            wx, wy, wz = imu_data[3:6]

            # Orientation Update
            # Create angular velocity vector in rover frame
            angluar_velocity = np.array([wx, wy, wz])
            diff_angle = angluar_velocity * dt

            # Get current rotation matrix from rover2world transform
            current_rotation = self.rover2world[0:3, 0:3]

            # Convert acceleration from rover frame to world frame
            acc_rover = np.array([ax, ay, az])
            acc_world = current_rotation @ acc_rover

            # Subtract lunar gravity in world frame
            acc_corrected = acc_world - self.Get
            
            # Integrate accleration to update velocity 
            self.vx = linear_speed + acc_corrected[0] * dt
            self.vy = acc_corrected[1] * dt
            self.vz = acc_corrected[2] * dt

            # Calculate displacement from velocity 
            displacement = np.array([self.vx, self.vy, self.vz])

            new2rover = pt.transform_from(
                pt.matrix_from_euler(
                    [diff_angle[2], diff_angle[1], diff_angle[0]], 2, 1, 0, extrinsic=False
                ),
                displacement,
            )

            # Apply rotation first (in rover frame), then translation (in world frame)
            self.rover2world = self.rover2world @ new2rover

            new_position, new_orientation = self._get_odom_from_transformation3d(
                self.rover2world
            )

            return new_position, new_orientation





