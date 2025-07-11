import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from std_msgs.msg import Float32
import argparse

"""
< This node subscribes image and publishes segmented images by using unet. >

1. Subscribe image
2. Input preprocessed image into the unet model
3. Classify into 6 classes for each pixels
4. Create segmented colored image
5. Create mono image
6. Publish both bgr image and mono image
"""
class ImageSegmentationNode(Node):
    def __init__(self, model_type='vgg16'):
        super().__init__('image_segmentation_node')

        # Load the model
        model_path = 'model/vgg16.h5' if model_type.lower() == 'vgg16' else 'model/mobilenet_1.h5'
        self.model = tf.keras.models.load_model(model_path)
        
        self.bridge = CvBridge()

        self.gt_mask = None

        # Subsvribe image
        self.subscription_image = self.create_subscription(
            Image,
            # ================== Change this topic to active camera topic ====================
            '/frontleft_camera/image', 
            self.image_callback,
            10
        )

        # Subscribe ground truth mask
        self.subscription_mask = self.create_subscription(
            Image,
            '/frontleft_camera/ground_truth',
            self.mask_callback,
            10
        )

        # Publisher for orb slam3
        self.publisher_image = self.create_publisher(
            Image, '/segmented_image', 10
        )
        
        # Publisher for visualization
        self.publisher_mask = self.create_publisher(
            Image, '/segmented_mask', 10
        )

        # Publisher F1 score
        self.publisher_f1_score = self.create_publisher(
            Float32, '/f1_score', 10
        )

    def mask_callback(self, msg):
        # Convert ros2 image to opencv image
        groundtruth_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self.gt_mask = cv2.resize(groundtruth_mask, (480, 480))
        self.get_logger().info("Ground truth mask received")
        

    def image_callback(self, msg):
        try:
            # Input size
            H = 480
            W = 480
            num_classes = 6
            
            # Convert input image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            # Preprocess image (Resize and Normalize)
            img = cv2.resize(cv_image, (W, H))
            img = img / 255.0
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)  # Expand (1, 480, 480, 3)

            # Predict
            pred_mask = self.model.predict(img)
            pred_mask = np.argmax(pred_mask, axis=-1)[0]

            # Based on LAC semantic camera
            color_map = {
                0: [0, 0, 0],        # Rover O.K
                1: [81, 0, 81],      # Ground O.K
                2: [108, 59, 42],    # Rocks O.K
                3: [70, 130, 180],   # Earth
                4: [110, 190, 160],  # Lander O.K
                5: [250, 170, 30]    # Fiducials O.K
            }

            # Create an empty color image (480x480, 3 channels)
            segmented_mask = np.zeros((H, W, 3), dtype=np.uint8)

            # Create semantic image
            for cls in range(num_classes):
                segmented_mask[pred_mask == cls] = color_map[cls]

            # Publish segmented mask
            segmented_msg = self.bridge.cv2_to_imgmsg(segmented_mask, encoding='bgr8')
            segmented_msg.header.stamp = self.get_clock().now().to_msg()
            segmented_msg.header.frame_id = 'camera'
            self.publisher_mask.publish(segmented_msg)

            # Publish gray scale image
            segmented_image = pred_mask.astype(np.uint8)
            segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='mono8')
            segmented_image_msg.header.stamp = self.get_clock().now().to_msg()
            segmented_image_msg.header.frame_id = 'camera'
            self.publisher_image.publish(segmented_image_msg)

            
            if self.gt_mask is not None:
                mapped_gt_mask = self.map_gt_values(self.gt_mask)
                pred_flat = pred_mask.flatten()
                gt_flat = mapped_gt_mask.flatten()
                score = f1_score(gt_flat, pred_flat, average='micro')

                self.get_logger().info(f'F1 Score: {score:.4f}')

                # Publish F1 score
                f1_msg = Float32()
                f1_msg.data = float(score)
                self.publisher_f1_score.publish(f1_msg)
            

            # Logger
            self.get_logger().info('Segmented image published')

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {str(e)}')
        except ValueError as e:
            self.get_logger().error(f'Value error (likely shape mismatch): {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {str(e)}')
    
    def map_gt_values(self, gt_mask):

        value_to_class = {
            0: 0,
            33: 1,
            60: 2,
            138: 3,
            137: 4,
            172: 5
        }

        mapped = np.zeros_like(gt_mask, dtype=np.uint8)
        for val, cls in value_to_class.items():
            mapped[gt_mask == val] = cls
        return mapped
    
        

def main(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vgg16', choices=['vgg16', 'mobilenet'])
    args = parser.parse_args(rclpy.get_default_context().argv[1:])

    rclpy.init(args=None)
    node = ImageSegmentationNode(model_type=args.model)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()