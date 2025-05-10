#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import carla

"""
This node is used in my_agent.py.
This node publishes the grayscale image and the semantic image from active camera. 
"""

class ROS2Publisher(Node):
    def __init__(self, sensors):
        super().__init__('ros2_publisher')

        # Define CvBridge object
        self.bridge = CvBridge()
        self.sensors = sensors

        # Publishers
        self.pub_dict = {}
        for position, config in sensors.items():

            if config['camera_active']:

                # Publishers for each active camera
                self.pub_dict[position] = {
                    'grayscale': self.create_publisher(Image, f'/{position.name.lower()}_camera/image', 10),
                    'semantic': self.create_publisher(Image, f'/{position.name.lower()}_camera/ground_truth', 10)
                }

        self.get_logger().info(f'ROS2Publisher initialized for {len(self.pub_dict)} active cameras')

    def publish_images(self, grayscale_data, semantic_data):
        timestamp = self.get_clock().now().to_msg()

        for position in self.pub_dict:
            # Grayscale image
            grayscale_img = grayscale_data.get(position, None)
            if grayscale_img is not None:
                img = grayscale_img.astype(np.uint8)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                ros_img = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
                ros_img.header.stamp = timestamp
                self.get_logger().info(f'Publishing grayscale for {position.name}')
                self.pub_dict[position]['grayscale'].publish(ros_img)

            # Semantic segmentation image
            semantic_img = semantic_data.get(position, None)
            if semantic_img is not None:
                semantic_img = semantic_img.astype(np.uint8)
                ros_semantic_img = self.bridge.cv2_to_imgmsg(semantic_img, encoding="rgb8")
                ros_semantic_img.header.stamp = timestamp
                self.get_logger().info(f'Publishing ground truth mask for {position.name}')
                self.pub_dict[position]['semantic'].publish(ros_semantic_img)