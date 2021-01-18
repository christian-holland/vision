import os
from PIL import Image
import torch
from train import get_transform
import torchvision
from inference_utils import draw_coco_box, draw_kps
import matplotlib.pyplot as plt
import numpy as np
import cv2


IS_KEYPOINT_NET = True
# source_dir = "/home/logi/datasets/forklift_coco/val2017"
source_dir = "/home/logi/datasets/2020_1222_brummer_dec_color/JPEGImages_Segmentations"
target_dir = "output_keypoints_dect"

allowed_types = [".png", ".jpg", ".jpeg"]
if IS_KEYPOINT_NET:
    model_type = "keypointrcnn_resnet50_fpn"
    load_path = "/home/logi/repos/forklift_rcnn/vision/references/detection/model.pth"
    kwargs = {"num_keypoints":6}
    num_classes = 2
else:
    model_type = "fasterrcnn_resnet50_fpn"
    load_path = "/home/logi/repos/forklift_rcnn/vision/references/detection/model_49.pth"
    num_classes = 3
    kwargs = {}

score_th = 0.9


if not os.path.exists(target_dir):
    os.makedirs(target_dir)

file_names = [imname for imname in os.listdir(source_dir)
              if any([imname.lower().endswith(dtype) for dtype in allowed_types])]

transform = get_transform(train=False)
model = torchvision.models.detection.__dict__[model_type](num_classes=num_classes, pretrained=False, **kwargs)
checkpoint = torch.load(load_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.cuda()
model.eval()

for image_name in file_names:
    image_path = os.path.join(source_dir, image_name)
    print("imname", image_name)
    image = Image.open(image_path)
    im = image.copy()
    im, _ = transform(im, None)
    im = im.unsqueeze(0)
    im = im.cuda()
    outputs = model(im)[0]
    prop_inds = torch.where(outputs["scores"] > score_th)
    print(prop_inds)
    scores =outputs["scores"][prop_inds]
    boxes = outputs["boxes"][prop_inds]
    labels = outputs["labels"][prop_inds]
    if IS_KEYPOINT_NET:
        keypoints = outputs["keypoints"][prop_inds][:, :, :2]
        kp_scores = outputs["keypoints_scores"][prop_inds]
    image = cv2.imread(image_path)
    for i in range(len(scores)):
        label = labels[i]
        box = boxes[i]
        image = draw_coco_box(image, box, str(label))
        if IS_KEYPOINT_NET:
            image = draw_kps(image, keypoints[i], kp_scores[i])
    # if len(prop_inds[0]):
    #     cv2.imwrite(os.path.join(target_dir, image_name), image)
    plt.imshow(image[:,:,::-1])
    plt.show()




