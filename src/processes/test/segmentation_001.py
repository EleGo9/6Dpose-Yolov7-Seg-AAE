import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

from matplotlib import pyplot as plt

learning_rate = 1e-4
batch_size = 4
image_size = 224
epochs = 10
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.Resize((600, 600)),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)]
     )
target_transform = transforms.Compose(
    [
        transforms.Resize((600, 600), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ]
)

train_pascal_voc = VOCSegmentation(
    root=".",
    year="2007",
    image_set="train",
    download=True,
    transform=transform,
    target_transform=target_transform
    )
train_loader = DataLoader(
    train_pascal_voc,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
    )

test_pascal_voc = VOCSegmentation(
    root=".",
    year="2007",
    image_set="test",
    download=True,
    transform=transform,
    target_transform=target_transform
    )
test_loader = DataLoader(
    test_pascal_voc,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
    )
classes = (
    "person",
    "bird", "cat", "cow", "dog", "horse", "sheep",
    "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    "bottle", "chair", "dining table", "potted plant", "sofa", "tv/monitor"
    )

custom_mask_rcnn = maskrcnn_resnet50_fpn(pretrained=True)
in_features = custom_mask_rcnn.roi_heads.box_predictor.cls_score.in_features 
custom_mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
custom_mask_rcnn.to(device)

optimizer = torch.optim.AdamW(params=custom_mask_rcnn.parameters(), lr=learning_rate)
custom_mask_rcnn.train()

def loadData():
  batch_Imgs=[]
  batch_Data=[]
  for i in range(batchSize):
        idx=random.randint(0,len(imgs)-1)
        img = cv2.imread(os.path.join(imgs[idx], "Image.jpg"))
        img = cv2.resize(img, imageSize, cv2.INTER_LINEAR)
        maskDir=os.path.join(imgs[idx], "Vessels")
        masks=[]
        for mskName in os.listdir(maskDir):
            vesMask = cv2.imread(maskDir+'/'+mskName, 0)
            vesMask = (vesMask > 0).astype(np.uint8) 
            vesMask=cv2.resize(vesMask,imageSize,cv2.INTER_NEAREST)
            masks.append(vesMask)
        num_objs = len(masks)
        if num_objs==0: return loadData()
        boxes = torch.zeros([num_objs,4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        img = torch.as_tensor(img, dtype=torch.float32)
        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   
        data["masks"] = masks
        batch_Imgs.append(img)
        batch_Data.append(data)  
  
  batch_Imgs=torch.stack([torch.as_tensor(d) for d in batch_Imgs],0)
  batch_Imgs = batch_Imgs.swapaxes(1, 3).swapaxes(2, 3)
  return batch_Imags, batch_Data

# for e in range(epochs):
for i, (images, segmentations) in enumerate(train_loader):
  plt.imshow(segmentations[0].squeeze(0).numpy())
  plt.show()
  break
  images, segmentations = images.to(device), segmentations.to(device)
  print(images.shape, segmentations.shape)

  data_targets = []
  bounding_boxes = torch.zeros([batch_size, 4], dtype=torch.float32)
  for i in range(batch_size):
    x, y, w, h = cv2.boundingRect(segmentations[i])
    bounding_boxes[i] = torch.tensor([x, y, x+w, y+h])
    target = {}
    target["bounding_boxes"] = bounding_boxes
    target["labels"] = torch.ones((batch_size,), dtype=torch.int64)   
    target["masks"] = segmentations
    data_targets.append(target)

  targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

  optimizer.zero_grad()
  loss_dict = custom_mask_rcnn(x, y)
  losses = sum(loss for loss in loss_dict.values())
  losses.backward()
  optimizer.step()

  print(i,'loss:', losses.item())
  if i == 2:
    break
  else:
    if i%10==0:
      torch.save(custom_mask_rcnn.state_dict(), str(i)+".torch")
      print("Save model to:",str(i)+".torch")