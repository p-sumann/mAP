import torch

def convert_midpoint_to_corner(boxes):
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack((x1, y1, x2, y2), axis=-1)

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    
    if box_format == "midpoint":
        boxes_preds = convert_midpoint_to_corner(boxes_preds)
        boxes_labels = convert_midpoint_to_corner(boxes_labels) 
    
    # n, 4 where n is number of boxes
    box1_x1 = boxes_preds[..., 0:1] 
    box1_y1 = boxes_preds[..., 1:2]
    box1_x2 = boxes_preds[..., 2:3]
    box1_y2 = boxes_preds[..., 3:4]
    box2_x1 = boxes_preds[..., 0:1] 
    box2_y1 = boxes_preds[..., 1:2]
    box2_x2 = boxes_preds[..., 2:3]
    box2_y2 = boxes_preds[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area  = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    
    iou = intersection / (box1_area + box2_area - intersection)
    
    return iou
    
    
    
    
    
    