from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
import os
import numpy as np
from matplotlib import pyplot as plt
# モデルのロード
model = load_model('models/LunarModel_2.h5')

# 予測関数
def predict_image(img_path, mask_path, model):
    H = 480
    W = 480
    num_classes = 4

    img = imread(img_path)
    img = img[:480, :480, :]
    img = img / 255.0
    img = img.astype(np.float32)

    mask = imread(mask_path, as_gray=True)
    mask = mask[:480, :480]

    pred_mask = model.predict(np.expand_dims(img, axis=0), verbose=0)
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]

    inter = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)
    iou = inter.sum() / union.sum()

    return img, mask, pred_mask, iou

# 画像とマスクのディレクトリ
img_dir = os.path.expanduser("~/Lunar_Autonomy_Challenge/archive/images/render/")
mask_dir = os.path.expanduser("~/Lunar_Autonomy_Challenge/archive/images/clean/")

# ファイルリストを取得
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
assert len(img_files) == len(mask_files), "画像とマスクの数が一致しません！"

# IoUスコアを保存するリスト
iou_scores = []

# 表示用のデータを保存（最初の5枚のみ）
display_results = []
num_display = 5

# すべての画像を処理
for i, (img_file, mask_file) in enumerate(zip(img_files, mask_files)):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)
    img, mask, pred_mask, iou = predict_image(img_path, mask_path, model)
    
    # IoUスコアを保存
    iou_scores.append(iou)
    
    # 最初の5枚だけ表示用に保存
    if i < num_display:
        display_results.append((img, mask, pred_mask, iou, img_file))
    
    # メモリ解放（必要なら）
    del img, mask, pred_mask

# 表示（最初の5枚）
fig, axes = plt.subplots(num_display, 3, figsize=(15, 5 * num_display))
for i, (img, mask, pred_mask, iou, img_file) in enumerate(display_results):
    axes[i, 0].set_title(f"Input Image: {img_file}")
    axes[i, 0].imshow(img)
    axes[i, 0].axis('off')
    axes[i, 1].set_title("True Mask")
    axes[i, 1].imshow(mask)
    axes[i, 1].axis('off')
    axes[i, 2].set_title(f"Predicted Mask (IoU: {iou:.2f})")
    axes[i, 2].imshow(pred_mask)
    axes[i, 2].axis('off')
plt.tight_layout()
plt.show()

# IoUスコアの分布を可視化
plt.figure(figsize=(10, 6))

# ヒストグラム
plt.subplot(1, 2, 1)
plt.hist(iou_scores, bins=20, color='skyblue', edgecolor='black')
plt.title('IoU Score Distribution')
plt.xlabel('IoU Score')
plt.ylabel('Frequency')
plt.grid(True)

# ボックスプロット
plt.subplot(1, 2, 2)
plt.boxplot(iou_scores, vert=False)
plt.title('IoU Score Boxplot')
plt.xlabel('IoU Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# 統計情報の表示
print(f"画像数: {len(iou_scores)}")
print(f"平均IoU: {np.mean(iou_scores):.3f}")
print(f"中央値IoU: {np.median(iou_scores):.3f}")
print(f"最小IoU: {np.min(iou_scores):.3f}")
print(f"最大IoU: {np.max(iou_scores):.3f}")