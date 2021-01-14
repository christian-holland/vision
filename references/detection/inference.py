import os
from PIL import Image
import torch
from train import get_transform
import torchvision
from inference_utils import draw_coco_box
import matplotlib.pyplot as plt
import numpy as np
import cv2

source_dir = "/home/logi/datasets/forklift_coco/val2017"
target_dir = "output"
allowed_types = [".png", ".jpg", ".jpeg"]
model_type = "fasterrcnn_resnet50_fpn"
score_th = 0.9
load_path = "/home/logi/repos/forklift_rcnn/vision/references/detection/model_49.pth"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

file_names = [imname for imname in os.listdir(source_dir)
              if any([imname.lower().endswith(dtype) for dtype in allowed_types])]

transform = get_transform(train=False)
model = torchvision.models.detection.__dict__[model_type](num_classes=3, pretrained=False)
checkpoint = torch.load(load_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.cuda()
model.eval()

for image_name in file_names:
    image_path = os.path.join(source_dir, image_name)
    image = Image.open(image_path)
    im = image.copy()
    im, _ = transform(im, None)
    im = im.unsqueeze(0)
    im = im.cuda()
    outputs = model(im)[0]
    prop_inds = torch.where(outputs["scores"] > score_th)
    scores =outputs["scores"][prop_inds]
    boxes = outputs["boxes"][prop_inds]
    labels = outputs["labels"][prop_inds]
    image = cv2.imread(image_path)
    for i in range(len(scores)):
        label = labels[i]
        box = boxes[i]
        image = draw_coco_box(image, box, str(label))
    plt.imshow(image[:,:,::-1])
    plt.show()




