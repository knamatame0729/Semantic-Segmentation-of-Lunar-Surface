import tensorflow as tf

# Check the device that tensorflow is using
physical_devices = tf.config.list_physical_devices('GPU')
logical_devices = tf.config.list_logical_devices('GPU')

# Check if GPU is available
print(f"Physical devices: {physical_devices}")
print(f"Logical devices: {logical_devices}")

