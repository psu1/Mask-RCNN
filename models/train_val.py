'''
Train and Val epoch of Mask-RCNN
'''
import os
import re

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from models.lossFunction import compute_losses

from tools.log import log, printProgressBar
import tools.visualize as visualize
from utils.dataLoader import Dataset
from utils.mask_rcnn_utils import mold_inputs, unmold_detections


def find_last(model):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        log_dir: The directory where events and weights are saved
        checkpoint_path: the path to the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model.model_dir))[1]
    key = model.config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return None, None
    # Pick last directory
    dir_name = os.path.join(model.model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint

def set_trainable(model, layer_regex, indent=0, verbose=1):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """

    for param in model.named_parameters():
        layer_name = param[0]
        trainable = bool(re.fullmatch(layer_regex, layer_name))
        if not trainable:
            param[1].requires_grad = False

def train_model(model, train_dataset, val_dataset, learning_rate, epochs, layers):
    """Train the model.
    train_dataset, val_dataset: Training and validation Dataset objects.
    learning_rate: The learning rate to train with
    epochs: Number of training epochs. Note that previous training epochs
            are considered to be done alreay, so this actually determines
            the epochs to train in total rather than in this particaular
            call.
    layers: Allows selecting wich layers to train. It can be:
        - A regular expression to match layer names to train
        - One of these predefined values:
          heaads: The RPN, classifier and mask heads of the network
          all: All the layers
          3+: Train Resnet stage 3 and up
          4+: Train Resnet stage 4 and up
          5+: Train Resnet stage 5 and up
    """

    # Pre-defined layer regular expressions
    layer_regex = {
        # all layers but the backbone
        "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # From a specific Resnet stage and up
        "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
        # All layers
        "all": ".*",
    }
    if layers in layer_regex.keys():
        layers = layer_regex[layers]

    # Data generators
    train_set = Dataset(train_dataset, model.config, augment=True)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    val_set = Dataset(val_dataset, model.config, augment=True)
    val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4)

    # Train
    log("\nStarting at epoch {}. LR={}\n".format(model.epoch + 1, learning_rate))
    log("Checkpoint Path: {}".format(model.checkpoint_path))
    set_trainable(model, layers)

    # Optimizer object
    # Add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': model.config.WEIGHT_DECAY},
        {'params': trainables_only_bn}
    ], lr=learning_rate, momentum=model.config.LEARNING_MOMENTUM)

    for epoch in range(model.epoch + 1, epochs + 1):
        log("Epoch {}/{}.".format(epoch, epochs))

        # Training
        loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask = train_epoch(
            model, train_generator, optimizer, model.config.STEPS_PER_EPOCH)

        # Validation
        val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask = valid_epoch(
            model, val_generator, model.config.VALIDATION_STEPS)

        # Statistics
        model.loss_history.append(
            [loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask])
        model.val_loss_history.append(
            [val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
             val_loss_mrcnn_mask])
        visualize.plot_loss(model.loss_history, model.val_loss_history, save=True, log_dir=model.log_dir)

        # Save model
        torch.save(model.state_dict(), model.checkpoint_path.format(epoch))

    model.epoch = epochs

def train_epoch(model, datagenerator, optimizer, steps):
    batch_count = 0
    loss_sum = 0
    loss_rpn_class_sum = 0
    loss_rpn_bbox_sum = 0
    loss_mrcnn_class_sum = 0
    loss_mrcnn_bbox_sum = 0
    loss_mrcnn_mask_sum = 0
    step = 0

    optimizer.zero_grad()

    for inputs in datagenerator:
        batch_count += 1

        images = inputs[0]  # [1,3,1024,1024]
        image_metas = inputs[1]  # [1, 89]
        rpn_match = inputs[2]  # [1,261888,1]
        rpn_bbox = inputs[3]  # [1,256,4]
        gt_class_ids = inputs[4]  # [1,23]
        gt_boxes = inputs[5]  # [1,23,4]
        gt_masks = inputs[6]  # [1,23,56,56]

        # image_metas as numpy array
        image_metas = image_metas.numpy()

        # Wrap in variables
        images = Variable(images)
        rpn_match = Variable(rpn_match)
        rpn_bbox = Variable(rpn_bbox)
        gt_class_ids = Variable(gt_class_ids)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        # To GPU
        if model.config.GPU_COUNT:
            images = images.cuda()
            rpn_match = rpn_match.cuda()
            rpn_bbox = rpn_bbox.cuda()
            gt_class_ids = gt_class_ids.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_masks = gt_masks.cuda()

        # Run object detection (call model forward pass)
        rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
            model([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

        # Compute losses
        rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match,
                                                                                                           rpn_bbox,
                                                                                                           rpn_class_logits,
                                                                                                           rpn_pred_bbox,
                                                                                                           target_class_ids,
                                                                                                           mrcnn_class_logits,
                                                                                                           target_deltas,
                                                                                                           mrcnn_bbox,
                                                                                                           target_mask,
                                                                                                           mrcnn_mask)
        loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        if (batch_count % model.config.BATCH_SIZE) == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_count = 0

        # Progress
        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                         suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                             loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                             mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                             mrcnn_mask_loss.data.cpu()[0]), length=10)

        # Statistics
        loss_sum += loss.data.cpu()[0] / steps
        loss_rpn_class_sum += rpn_class_loss.data.cpu()[0] / steps
        loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0] / steps
        loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0] / steps
        loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0] / steps
        loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0] / steps

        # Break after 'steps' steps
        if step == steps - 1:
            break
        step += 1

    return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum


def valid_epoch(model, datagenerator, steps):
    step = 0
    loss_sum = 0
    loss_rpn_class_sum = 0
    loss_rpn_bbox_sum = 0
    loss_mrcnn_class_sum = 0
    loss_mrcnn_bbox_sum = 0
    loss_mrcnn_mask_sum = 0

    for inputs in datagenerator:
        images = inputs[0]
        image_metas = inputs[1]
        rpn_match = inputs[2]
        rpn_bbox = inputs[3]
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        # image_metas as numpy array
        image_metas = image_metas.numpy()

        # Wrap in variables
        images = Variable(images, volatile=True)
        rpn_match = Variable(rpn_match, volatile=True)
        rpn_bbox = Variable(rpn_bbox, volatile=True)
        gt_class_ids = Variable(gt_class_ids, volatile=True)
        gt_boxes = Variable(gt_boxes, volatile=True)
        gt_masks = Variable(gt_masks, volatile=True)

        # To GPU
        if model.config.GPU_COUNT:
            images = images.cuda()
            rpn_match = rpn_match.cuda()
            rpn_bbox = rpn_bbox.cuda()
            gt_class_ids = gt_class_ids.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_masks = gt_masks.cuda()

        # Run object detection (call model forward pass)
        rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
            model([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

        if target_class_ids.size() == torch.Size([0]):  # changed by jaden
            continue

        # Compute losses
        rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match,
                                                                                                           rpn_bbox,
                                                                                                           rpn_class_logits,
                                                                                                           rpn_pred_bbox,
                                                                                                           target_class_ids,
                                                                                                           mrcnn_class_logits,
                                                                                                           target_deltas,
                                                                                                           mrcnn_bbox,
                                                                                                           target_mask,
                                                                                                           mrcnn_mask)
        loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

        # Progress
        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                         suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f}".format(
                             loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                             mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                             mrcnn_mask_loss.data.cpu()[0]), length=10)

        # Statistics
        loss_sum += loss.data.cpu()[0] / steps
        loss_rpn_class_sum += rpn_class_loss.data.cpu()[0] / steps
        loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0] / steps
        loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0] / steps
        loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0] / steps
        loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0] / steps

        # Break after 'steps' steps
        if step == steps - 1:
            break
        step += 1

    return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum


def detect(model, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = mold_inputs(model.config, images)

        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        # To GPU
        if model.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Wrap in variable
        molded_images = Variable(molded_images, volatile=True)

        # Run object detection
        detections, mrcnn_mask = model([molded_images, image_metas], mode='inference')

        # Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results
