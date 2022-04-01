import torch
import torchvision
from torch import nn
import numpy as np

torch.manual_seed(0)

track_ids = torch.tensor([1,5,6,13,14,20,26,89])

box_features = torch.rand((100, 512))

# anchor_list = torch.rand((100,4))
# gt_box = torch.rand((7,4))

iou_scores = torch.rand((100,7))

iou_max, index = iou_scores.max(1)

id_index = iou_max > 0.9

index = index[id_index]

embedding = box_features[id_index]
tids = track_ids[index]

print('')
