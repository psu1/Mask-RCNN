"""
Mask R-CNN Training Code
"""
import argparse
from models.config import cfg, set_cfg_value, cfg_from_file
from models.mask_rcnn import  MaskRCNN
from models.train_val import *

from utils.coco import CocoDataset, evaluate_coco
from tools.fprintfLog import fprintf_log
import datetime

import pprint

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")

    parser.add_argument('--cfg', '-c', dest='cfg_file', required=False,
                        help='Config file to run')

    parser.add_argument('--last', required=False, type=bool,
                        default= False,
                        help="Continue training the last model you trained")

    return parser.parse_args()

def main():
    args = parse_args()

    # Configuration
    cfg_from_file(args.cfg_file)
    # Adaptively adjust some configs
    set_cfg_value()
    print(pprint.pformat(cfg))

    # Create model
    model = MaskRCNN(config=cfg)
    if cfg.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.last:
        model_path = find_last(model)[1]
        print('Continue training on {}'.format(model_path))
    else:
        model_path = os.path.join(cfg.ROOT_DIR, cfg.TRAIN.WEIGHTS)
        print("Loading weights ", model_path)
        model.load_weights(model_path)


    # save printing logs to file
    now = datetime.datetime.now()
    save_log_dir = os.path.join(cfg.TRAIN.LOG_DIR, "{}{:%Y%m%dT%H%M}".format(
        cfg.MODEL.NAME.lower(), now))

    # Train or evaluate
    if args.command == "train":

        # save config and model
        fprintf_log(pprint.pformat(cfg), save_log_dir)
        fprintf_log(model, save_log_dir)

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(cfg.DATASET.PATH, "train", year=cfg.DATASET.YEAR, auto_download=cfg.DATASET.DOWNLOAD)
        dataset_train.load_coco(cfg.DATASET.PATH, "valminusminival", year=cfg.DATASET.YEAR, auto_download=cfg.DATASET.DOWNLOAD)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(cfg.DATASET.PATH, "minival", year=cfg.DATASET.YEAR, auto_download=cfg.DATASET.DOWNLOAD)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        train_model(model,
                    dataset_train, dataset_val,
                    learning_rate=cfg.SOLVER.BASE_LR,
                    epochs=cfg.TRAIN.TRAIN_SCHEDULE[0],
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        train_model(model,
                    dataset_train, dataset_val,
                    learning_rate=cfg.SOLVER.BASE_LR,
                    epochs=cfg.TRAIN.TRAIN_SCHEDULE[1],
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        train_model(model,
                    dataset_train, dataset_val,
                    learning_rate=cfg.SOLVER.BASE_LR / 10,
                    epochs=cfg.TRAIN.TRAIN_SCHEDULE[2],
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco = dataset_val.load_coco(cfg.DATASET.PATH, "minival", year=cfg.DATASET.YEAR, return_coco=True,
                                     auto_download=cfg.DATASET.DOWNLOAD)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(cfg.TEST.NUM_IMG))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(cfg.TEST.NUM_IMG))
        evaluate_coco(model, dataset_val, coco, "segm", limit=int(cfg.TEST.NUM_IMG))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))

if __name__ == '__main__':
    main()

