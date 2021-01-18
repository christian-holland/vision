import cv2
import numpy as np

kp_map = ["RL", "RR", "FL", "FR", "LL", "LR"]
min_score = 15

def name_to_color(name):
    color = abs(hash(name))
    color = (color % 256 ** 3 // 256 ** 2), color % 256 ** 2 // 256, color % 256
    return color

def draw_coco_box(image, bb, instance_type):
    color = name_to_color(instance_type)
    args = (image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
    image = cv2.rectangle(*args)
    return image

def draw_kps(image, keypoints, scores):
    keypoints = keypoints.long()
    keypoints = [tuple(list(kps)) for kps in keypoints]
    print("scores", [float(score) for score in scores])
    for kp, score, text in zip(keypoints, scores, kp_map):
        if score > min_score:
            image = cv2.circle(image, kp , radius=0, color=(0 , 0, 255), thickness=7)
            image = write(image, kp, text)
    return image


def write(image, pos, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    cv2.putText(image, text,
                pos,
                font,
                fontScale,
                fontColor,
                lineType)
    return image


if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    image = cv2.imread("/home/logi/datasets/forklift_coco/train2017/img_2020_12_22_00_03_05t107331.jpg")

    image = draw_kps(image, [[100, 100]], [0.5])
    plt.imshow(image[:,:,::-1])
    plt.show()