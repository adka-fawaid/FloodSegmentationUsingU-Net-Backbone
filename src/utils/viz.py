import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_error_map(pred, mask):
    """Binary XOR sebagai error map"""
    pred_bin = (pred > 0.5).astype(np.uint8)
    mask_bin = (mask > 0.5).astype(np.uint8)
    error = cv2.absdiff(pred_bin, mask_bin) * 255
    return error

def overlay_iou(img, iou, title):
    """Tulis IoU di atas gambar"""
    img = img.copy()
    text = f"{title} IoU: {iou:.4f}"
    cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)
    return img

def save_visualization(original, mask, pred, save_path, model_name, iou):
    """
    original: HxWx3
    mask: HxW
    pred: HxW
    """

    pred_img = (pred * 255).astype(np.uint8)
    mask_img = (mask * 255).astype(np.uint8)

    error = compute_error_map(pred, mask)

    pred_disp = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
    mask_disp = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    error_disp = cv2.cvtColor(error, cv2.COLOR_GRAY2BGR)

    pred_disp = overlay_iou(pred_disp, iou, model_name)

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 5, 1); plt.imshow(original); plt.axis("off"); plt.title("Original")
    plt.subplot(1, 5, 2); plt.imshow(mask_img, cmap="gray"); plt.axis("off"); plt.title("Mask GT")
    plt.subplot(1, 5, 3); plt.imshow(pred_img, cmap="gray"); plt.axis("off"); plt.title(f"{model_name} Pred")
    plt.subplot(1, 5, 4); plt.imshow(error_disp); plt.axis("off"); plt.title("Error Map")
    plt.subplot(1, 5, 5); plt.imshow(pred_disp); plt.axis("off"); plt.title(f"{model_name} IoU")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
