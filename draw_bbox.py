import cv2
import matplotlib.pyplot as plt

# =========================
# 输入
# =========================

image_path = "data/truthfulness/lesion_localization/images/image_1.png"

# 预测框
pred_bbox = [0.32, 0.35, 0.48, 0.55]

# GT框
gt_bbox = [0.51778515625, 0.476578125, 0.644033203125, 0.552109375]

# =========================
# 读取图片
# =========================

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape

# =========================
# 归一化坐标 -> 像素坐标
# =========================

def norm_to_pixel(box, w, h):
    x1, y1, x2, y2 = box

    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    return x1, y1, x2, y2

pred_box = norm_to_pixel(pred_bbox, w, h)
gt_box = norm_to_pixel(gt_bbox, w, h)

# =========================
# 画 GT 框（绿色）
# =========================

cv2.rectangle(
    image,
    (gt_box[0], gt_box[1]),
    (gt_box[2], gt_box[3]),
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    "GT",
    (gt_box[0], gt_box[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

# =========================
# 画 Prediction 框（红色）
# =========================

cv2.rectangle(
    image,
    (pred_box[0], pred_box[1]),
    (pred_box[2], pred_box[3]),
    (255, 0, 0),
    2
)

cv2.putText(
    image,
    "Prediction",
    (pred_box[0], pred_box[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 0, 0),
    2
)

# =========================
# 显示结果
# =========================

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.show()