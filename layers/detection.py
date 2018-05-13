'''
Detection Target Layer

Detection Layer

'''
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable


import utils.utils as utils
from lib.nms.nms_wrapper import nms
from lib.roialign.roi_align.crop_and_resize import CropAndResizeFunction
from utils.dataFormat import parse_image_meta
from layers.proposal_layer import apply_box_deltas
from tools.py_utils import unique1d, intersect1d


#  Detection Target Layer

def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
    boxes2 = boxes2.repeat(boxes2_repeat,1)

    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:,0] + b2_area[:,0] - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps

def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    # Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.

    if torch.nonzero(gt_class_ids < 0).size() != torch.Size([0]):
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix.data, :]
        crowd_masks = gt_masks[crowd_ix.data, :, :]
        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]
        gt_masks = gt_masks[non_crowd_ix.data, :]

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool =  Variable(torch.ByteTensor(proposals.size()[0]*[True]), requires_grad=False)
        if config.GPU_COUNT:
            no_crowd_bool = no_crowd_bool.cuda()

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    # Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    if torch.nonzero(positive_roi_bool).size() != torch.Size([0]):
        # pdb.set_trace()
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        #================should it be sorted by IOU?==========
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data,:]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data,:]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        # Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data,:,:]

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()
        masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)
    else:
        positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).size() != torch.Size([0]) and positive_count>0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count,4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count,4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()

    return rois, roi_gt_class_ids, deltas, masks



#  Detection Layer

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))

    return boxes

def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI
    # Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois *= scale

    # Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    # Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep_bool = class_ids>0

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    keep = torch.nonzero(keep_bool)[:,0]

    # Apply per-class NMS
    pre_nms_class_ids = class_ids[keep.data]
    pre_nms_scores = class_scores[keep.data]
    pre_nms_rois = refined_rois[keep.data]

    for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
        # Pick detections of this class
        ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]

        # Sort
        ix_rois = pre_nms_rois[ixs.data]
        ix_scores = pre_nms_scores[ixs]
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data,:]

        class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)

        # Map indicies
        class_keep = keep[ixs[order[class_keep].data].data]

        if i==0:
            nms_keep = class_keep
        else:
            nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
    keep = intersect1d(keep, nms_keep)

    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1)), dim=1)

    return result


def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    # Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)

    return detections

