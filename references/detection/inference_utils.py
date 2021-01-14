import cv2

def name_to_color(name):
    color = abs(hash(name))
    color = (color % 256 ** 3 // 256 ** 2), color % 256 ** 2 // 256, color % 256
    return color

def draw_coco_box(image, bb, instance_type):
    color = name_to_color(instance_type)
    args = (image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)
    image = cv2.rectangle(*args)
    return image
