import os
import random
import skimage.io
import matplotlib.pyplot as plt

from models.config import cfg, set_cfg_value, cfg_from_file
from models.mask_rcnn import  MaskRCNN
import tools.visualize as visualize

import torch

from models.train_val import detect


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# load and inference default config
cfg_from_file('cfgs/coco_train.yaml')
set_cfg_value()


# Create model object.
model = MaskRCNN(config=cfg)
if cfg.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load( cfg.DEMO.WEIGHTS))


# Load a random image from the images folder
file_names = next(os.walk(cfg.DEMO.IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(cfg.DEMO.IMAGE_DIR, random.choice(file_names)))
# image = skimage.io.imread(os.path.join(cfg.DEMO.IMAGE_DIR, 'img5.jpg'))

# Run detection
results = detect(model, [image])

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
plt.show()