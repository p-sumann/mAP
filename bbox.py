from typing import NamedTuple
import torch



class BBox(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    

# pascal VOC format
# BoundingBox(x_min, y_min, x_max, y_max)
# also know as (x1,y1,x2,y2) (xyxy) format

class COCO_bbox(NamedTuple):
    x_min: float
    y_min: float
    width: float
    heigth: float

# coco format
# xywh

class CxCyWHBoundingBox(NamedTuple):
    cx:float
    cy:float
    width: float
    height:float
# cxcywh values of these cordiantes are normalized
# these are yolo format