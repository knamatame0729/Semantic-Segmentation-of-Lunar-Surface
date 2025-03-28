from skimage.io import imread
from tensorflow.keras.models import load_model
import os
import numpy as np
import matplotlib.pyplot as plt

""" This code compute mean of IoU for all images and predcted mask"""

# Load Model
model = load_model('models/LunarModel_2.h5')


# IOU List to store
iou_scores = []

def predict_image(img_path, mask_path, model):
    H = 480
    W = 480
    num_classes = 4

    # Images
    img = imread(img_path)
    img = img[:480, :480, :]
    img = img / 255.0
    img = img.astype(np.float32)

    # Read mask
    mask = imread(mask_path, as_gray=True)
    mask = mask[:480, :480]
    
    # Prediction
    pred_mask = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]
    
    # Compute IOU score
    inter = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    
    union_sum = union.sum()

    if union_sum == 0: 
        return 0.0
    
    iou = inter.sum() / union_sum
    if not np.isfinite(iou):
        return 0.0
        
    return iou

# Path 
base_path = os.path.expanduser("~/Lunar_Surface_Semantic_Segmentation/archive/images")
img_path = os.path.join(base_path, "render")
mask_path = os.path.join(base_path, "clean")


img_files = [f for f in os.listdir(img_path) if f.startswith('render') and f.endswith('.png')]
image_ids = [f.replace('render', '').replace('.png', '') for f in img_files]

img_paths = [os.path.join(img_path, f"render{id}.png") for id in image_ids]
mask_paths = [os.path.join(mask_path, f"clean{id}.png") for id in image_ids]


# Compute IOU
for img_path, mask_path in zip(img_paths, mask_paths):
    iou = predict_image(img_path, mask_path, model)
    
    if iou is not None:  
        iou_scores.append(iou)

# Compute mean and variance of IOU
mean_iou = np.mean(iou_scores) if iou_scores else np.nan
variance_iou = np.var(iou_scores) if iou_scores else np.nan

print(f"Checked {len(iou_scores)} images")
print(f"Mean of IoU: {mean_iou:.4f}")
print(f"Variance of IoU: {variance_iou :.4f}")

plt.figure(figsize=(10, 6))
plt.hist(iou_scores, bins=100)
plt.title('Distribution of IoU', fontsize=14)
plt.xlabel('IoU Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean of IoU = {mean_iou:.4f}')
plt.legend()
plt.show()