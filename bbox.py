from typing import NamedTuple
import torch

class BBox(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    

# pascal VOC format
# BoundingBox(x_min, y_min, x_max, y_max)
# also know as (x1,y1,x2,y2)
