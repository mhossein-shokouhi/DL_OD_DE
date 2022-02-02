# Import necessary libraries
import torch

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams["figure.figsize"] = [30,10]
plt.rcParams["figure.autolayout"] = True

# Define necessary functions
def calc_object_depth(obj_masks, depth_map):
    obj_num = obj_masks.shape[0]
    obj_depths = np.empty(obj_num)
    for i in range(obj_num):
        obj_depth_map = depth_map.copy()
        np.putmask(obj_depth_map, ~obj_masks[i], 0)
        obj_depths[i] = np.true_divide(obj_depth_map.sum(),(obj_depth_map!=0).sum())
    return obj_depths

def create_labels(classes, depths, class_names):
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if depths is not None:
        if labels is None:
            labels = [d for d in depths]
        else:
            labels = ["{} {}".format(l, d) for l, d in zip(labels, depths)]
    return labels

# The main object detection function
def object_detection(input_img = "examples/267_image.jpg", depth_lim = 0, lim_type = 0):
    im = mpimg.imread(input_img)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    torch.cuda.empty_cache()

    # bring outputs back to CPU
    outputs_instances = outputs["instances"].to("cpu")
    depth_map = np.load("depth.npy")

    # prepare depth numbers (in meters)
    obj_depths = calc_object_depth(outputs_instances.pred_masks, depth_map)

    if (lim_type==1):
        idx = []
        for i in range(outputs_instances.pred_masks.shape[0]):
            if(obj_depths[i]<=depth_lim):
                idx.append(i)
        obj_depths = obj_depths[idx]
        outputs_instances = outputs_instances[idx]
    elif (lim_type==2):
        idx = []
        for i in range(outputs_instances.pred_masks.shape[0]):
            if(obj_depths[i]>=depth_lim):
                idx.append(i)
        obj_depths = obj_depths[idx]
        outputs_instances = outputs_instances[idx]

    obj_depths_str = [('{:.2f}'.format(x) + ' m') for x in obj_depths]

    # visualize the output
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    boxes = outputs_instances.pred_boxes if outputs_instances.has("pred_boxes") else None
    scores = outputs_instances.scores if outputs_instances.has("scores") else None
    classes = outputs_instances.pred_classes.tolist() if outputs_instances.has("pred_classes") else None
    labels = create_labels(classes, obj_depths_str, v.metadata.get("thing_classes", None))
    keypoints = outputs_instances.pred_keypoints if outputs_instances.has("pred_keypoints") else None

    masks = np.asarray(outputs_instances.pred_masks)
    masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

    colors = None
    alpha = 0.5

    v.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    cv2.imwrite("output.jpg", cv2.cvtColor(v.output.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR))

    class_names = v.metadata.get("thing_classes", None)
    with open("output.txt", "w") as filehandle:
        for i in range(outputs_instances.pred_masks.shape[0]):
            filehandle.write("Class: {} - Distance: {}\n".format(class_names[classes[i]], obj_depths_str[i]))
