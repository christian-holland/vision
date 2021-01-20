import torch
import random
import torchvision.transforms as T
from torchvision.transforms import functional as F
import albumentations as A
import numpy as np
import cv2

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

def _flip_forklift_keypoints(kps, width):
    flip_inds = [1, 0, 3, 2, 5, 4]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_forklift_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def prepare_annotations(keypoints):
    prepared_keypoints = []
    kp_inds = []
    instance_inds = []
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            if keypoints[i, j, 2]:
                prepared_keypoints.append(
                    (keypoints[i, j, 0], keypoints[i, j, 1]))
                instance_inds.append(str(i))
                kp_inds.append(str(j))
    return prepared_keypoints, kp_inds, instance_inds


def unprepare_annotations(keypoints, masks, labels, kp_id,
                          instance_id):
    masks = np.array(masks)
    num_kps = 6
    num_instances = labels.shape[0]
    kps_serial = np.zeros((num_instances, num_kps, 3), dtype=np.float)
    for i in range(len(keypoints)):
        # check if keypoints are within mask
        if masks[int(instance_id[i]), int(keypoints[i][1]), int(keypoints[i][0])] != 0:
            kps_serial[int(instance_id[i]), int(kp_id[i]), :2] = keypoints[i]
            kps_serial[int(instance_id[i]), int(kp_id[i]), 2] = 2
    valid_instances = ~np.all(kps_serial.reshape(kps_serial.shape[0], -1) == 0,
                              axis=1)
    kps_serial = kps_serial[valid_instances]
    labels = labels[valid_instances]
    masks = masks[valid_instances]
    bboxes = bbox_from_mask(masks)
    return kps_serial, masks, bboxes, labels


def bbox_from_mask(masks):
    bboxes = []
    for mask in masks:
        assert ~np.all(mask == 0)
        rows_nonzero = np.where(~np.all(mask == 0, axis=1))[0]
        cols_nonzero = np.where(~np.all(mask == 0, axis=0))[0]
        y_min = rows_nonzero[0]
        x_min = cols_nonzero[0]
        height = rows_nonzero[-1] + 1
        width = cols_nonzero[-1] + 1
        bboxes.append([x_min, y_min, width, height])
    if bboxes:
        return np.array(bboxes)
    else:
        return np.zeros((0, 4))




class AlbumentationTransforms(object):
    def __init__(self):
        pass

    def __call__(self, image, target):


        keypoints = target["keypoints"]
        masks = target["masks"]
        labels = target["labels"]


        image = np.array(image)
        masks = np.array(masks)

        # import matplotlib.pyplot as plt
        # import copy
        # from debug_utils import vis, visualize_bbox
        # before_labels = copy.deepcopy(labels)
        # before_bboxes = copy.deepcopy(bboxes)
        # before = vis(image, keypoints)

        transform = A.Compose(
            [   # A.imgaug.transforms.IAAPerspective(scale=(0.00, 0.05), keep_size=True, always_apply=False, p=0.5),
                # A.imgaug.transforms.IAAAffine(translate_percent=(-0.1, 0.1), mode="constant", p=0.5),
                A.Rotate(limit=180, p=1., border_mode=cv2.BORDER_CONSTANT),
                # A.RandomResizedCrop(h, w, scale=(0.8, 1.0), ratio=(w/float(h), w/float(h)), p=0.5),
            ],
            keypoint_params=A.KeypointParams(format='xy',
                                             label_fields=["kp_id",
                                                           "instance_id"])
        )
        prepared_kps, kp_id, instance_id = prepare_annotations(
            keypoints)
        transformed = transform(image=image,
                                keypoints=prepared_kps,
                                masks=masks,
                                # bboxes=bboxes,
                                # bbox_params=A.BboxParams(format="coco"),
                                kp_id=kp_id,
                                instance_id=instance_id)

        # for i in range(len(masks)):
        #     plt.imshow(np.concatenate([masks[i], transformed["masks"][i]], axis=1))
        #     plt.show()

        keypoints, masks, bboxes, labels = unprepare_annotations(
            transformed["keypoints"],
            transformed["masks"],
            labels,
            transformed["kp_id"],
            transformed["instance_id"])


        # after = vis(transformed["image"], keypoints)
        # for i in range(len(before_bboxes)):
        #     before = visualize_bbox(before, before_bboxes[i], str(before_labels[i]))
        # for i in range(len(bboxes)):
        #     after = visualize_bbox(after, bboxes[i], str(labels[i]))
        # print(keypoints)
        # plt.imshow(np.concatenate([before, after], axis=1))
        # plt.show()

        target["image"] = torch.tensor(transformed["image"])
        target["masks"] = torch.tensor(masks)
        target["keypoints"] = torch.FloatTensor(keypoints)
        target["labels"] = labels
        target["boxes"] = torch.FloatTensor(bboxes)
        for i in range(len(target["area"])):
            target["area"][i] = 0
        return transformed["image"], target