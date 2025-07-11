from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
import os
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import gray2rgb

# Load Model
#model = load_model('models/LunarModel_2.h5')

model = load_model('model/mobilenet_1.h5')

# Display
model.summary()

# function to predict result 
def predict_image(img_path, mask_path, model):
    H = 480
    W = 480
    num_classes = 6

    img = imread(img_path)
    img = img[:480, :480, :]
    img = img / 255.0
    img = img.astype(np.float32)

    ## Read mask
    mask = imread(mask_path, as_gray = True)
    mask = mask[:480, :480]
    
    ## Prediction
    pred_mask = model.predict(np.expand_dims(img, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]
    
    
    # Compute IOU score
    inter = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    
    iou = inter.sum() / union.sum()

    return img, mask, pred_mask, iou

image_ids = ["08000", "01726", "01742"]

img_path = os.path.expanduser("~/ORBSLAM_Semantic_Mapping/LunarAutonomyChallenge/archive") # image
mask_path = os.path.expanduser("~/ORBSLAM_Semantic_Mapping/LunarAutonomyChallenge/archive") # mask

img_paths = [os.path.join(img_path, "raw", f"raw_image_{id}.png") for id in image_ids]
mask_paths = [os.path.join(mask_path, "semantic", f"semantic_image_{id}.png") for id in image_ids]


fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
    img, mask, pred_mask, iou = predict_image(img_path, mask_path, model)

    axes[i, 0].set_title("Input Image")
    axes[i, 0].imshow(img)
    axes[i, 0].axis("off")

    axes[i, 1].set_title("True Mask")
    axes[i, 1].imshow(mask)
    axes[i, 1].axis("off")

    axes[i, 2].set_title(f"\nPredicted Mask with IOU: {iou:.2f}")
    axes[i, 2].imshow(pred_mask)
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
