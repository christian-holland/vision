import cv2
import matplotlib.pyplot as plt

KEYPOINT_MODEL_RES = (768, 1024)
image = cv2.imread("/data/0.jpg")
plt.imshow(image)
plt.show()

vertical = image.shape[0] > image.shape[1]
target_shape = (
    (KEYPOINT_MODEL_RES[1], KEYPOINT_MODEL_RES[0])
    if vertical
    else (KEYPOINT_MODEL_RES[0], KEYPOINT_MODEL_RES[1])
)

image = cv2.resize(image, target_shape)
plt.imshow(image)
plt.show()