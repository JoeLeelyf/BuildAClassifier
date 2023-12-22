import torchvision
import torch

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True, num_classes=2)

x = [torch.rand(3, 320, 320)]
predictions = model(x)
print(predictions.shape)