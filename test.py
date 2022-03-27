# import torch
# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator
# # load a pre-trained model for classification and return
# # only the features
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# # FasterRCNN needs to know the number of
# # output channels in a backbone. For mobilenet_v2, it's 1280
# # so we need to add it here
# backbone.out_channels = 1280

# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))

# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be ['0']. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
#                                                 output_size=7,
#                                                 sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes=2,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

# predictions = model(x)
# print(predictions[0])

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

# x = torch.rand(512, 256, 7, 7)

# print(x.size())

# boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
# print(boxes)

# m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
# i = OrderedDict()
# i['feat1'] = torch.rand(1, 5, 64, 64)
# i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
# i['feat3'] = torch.rand(1, 5, 16, 16)
# # create some random bounding boxes
# boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
# print(boxes)
# # original image size, before computing the feature maps
# image_sizes = [(512, 512)]
# output = m(i, [boxes], image_sizes)
# print(output.shape)

fc6 = nn.Linear(12544, 1024)
fc7 = nn.Linear(1024, 1024)

x = torch.rand(512,256,7,7)
print(x.shape)
x = x.flatten(start_dim=1)
print(x.shape)
x = F.relu(fc6(x))
print(x.shape)
x = F.relu(fc7(x))
print(x.shape)
