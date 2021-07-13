import os
from PIL import Image
import torch
from train import get_transform
import torchvision
from inference_utils import draw_coco_box, draw_kps
from constanz import NUM_KPS
import matplotlib.pyplot as plt
import numpy as np
import cv2
from custom_models.keypoint_rcnn import keypointrcnn_resnet50_fpn

IS_KEYPOINT_NET = True

# source_dir = "/home/logi/datasets/forklift_coco/val2017"
# source_dir = "/home/logi/datasets/selected_data"
# target_dir = "data/im160*rotfullstep500_march_resized"
source_dir = "/home/logi/datasets/faurecia_coco/test"
target_dir = "/home/logi/datasets/faurecia_coco/inference_vis"
# target_dir = "data/im160*rotfullstep500_resized"

allowed_types = [".png", ".jpg", ".jpeg"]
if IS_KEYPOINT_NET:
    model_type = "keypointrcnn_resnet50_fpn"
    load_path = "runs/2021_06_30 19:37:08/model.pth"
    # load_path = "/home/logi/repos/forklift_rcnn/vision/references/detection/model120im_step50.pth"
    kwargs = {"num_keypoints": NUM_KPS}
    num_classes = 2
else:
    model_type = "fasterrcnn_resnet50_fpn"
    load_path = "/home/logi/repos/forklift_rcnn/vision/references/detection/model_49.pth"
    num_classes = 3
    kwargs = {}

score_th = 0.9
kp_score_th = 0



if not os.path.exists(target_dir):
    os.makedirs(target_dir)

file_names = []
for dirpath, dirnames, filenames in os.walk(source_dir):
    filenames = list(filenames)
    filenames = [imname for imname in filenames if any([imname.lower().endswith(dtype) for dtype in allowed_types])]
    for filename in filenames:
        file_names.append(os.path.join(dirpath, filename))


transform = get_transform(train=False)
model = keypointrcnn_resnet50_fpn(num_classes=num_classes, pretrained=False, **kwargs)
checkpoint = torch.load(load_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.cuda()
model.eval()

for image_path in file_names:
    image_name = image_path.split("/")[-1]
    print("imname", image_name)
    # image = Image.open(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    # image = cv2.imread(image_path)
    # image = cv2.resize(image, (768, 1024))
    for i in range(len(scores)):
        label = labels[i]
        box = boxes[i]
        image = draw_coco_box(image, box, str(label))
        if IS_KEYPOINT_NET:
            image = draw_kps(image, keypoints[i], kp_scores[i], kp_score_th)
    # if len(prop_inds[0]):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(target_dir, image_name), image)
    # plt.imshow(image[:,:,::-1])
    # plt.show()




