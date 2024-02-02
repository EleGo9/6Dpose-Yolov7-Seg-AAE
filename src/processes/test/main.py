import os
import sys

import torch
import numpy as np
import cv2
sys.path.append(os.getcwd())

from src.methods.dataset.custom_dataset import CustomDataset
from src.methods.dataloader.custom_dataloader import CustomDataloader
from src.methods.transform.custom_transform import CustomTransform

root = "dataset/root 14/data/01"
config_file_name = "dataset/images_labels.txt"
batch_size = 4
shuffle = True
num_workers = 2
num_epochs = 10
save_frequency = 5
show = True

learning_rate = 1e-3
image_size = (240, 320)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from torchvision.transforms import *
transform = CustomTransform(image_size)
transform_image = Compose([
    ToPILImage(),
    Resize(image_size),
    ToTensor()
])
transform_label = Compose([
    ToPILImage(),
    Grayscale(),
    Resize(image_size, InterpolationMode.NEAREST),
    ToTensor()
])
dataset = CustomDataset(
    config_file_name,
    root,
    images_dir="rgb/",
    labels_dir="mask/",
    transform_image=transform_image,
    transform_label=transform_label
)
dataloader = CustomDataloader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers
    #collate_fn=lambda x: list(zip(*x))
)

from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torch.nn import Conv2d
model = deeplabv3_resnet50(pre_trained=True)
model.classifier[4] = Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
model = model.to(device)
model.train()

from torch.optim import Adam
optimizer = Adam(params=model.parameters(), lr=learning_rate)

for e in range(num_epochs):
    for i, (index, image, label) in enumerate(dataloader):
        image = torch.autograd.Variable(image, requires_grad=False).to(device)
        label = torch.autograd.Variable(label, requires_grad=False).to(device).squeeze(1)

        prediction = model(image)["out"]
        model.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(prediction, label.long())
        loss.backward()
        optimizer.step()
        segmentation = torch.argmax(prediction[0], 0).cpu().detach().numpy()

        if show:
            image_show = image[0].cpu().detach().numpy().transpose(1, 2, 0)
            label_show = label[0].cpu().detach().numpy()
            label_show = np.dstack((label_show, np.zeros_like(label_show), np.zeros_like(label_show)))
            segmentation_show = np.where(segmentation == 1, 255, 0)
            segmentation_show = segmentation_show.astype(np.uint8)
            segmentation_show = np.dstack((segmentation_show, np.zeros_like(segmentation_show), np.zeros_like(segmentation_show)))
            full_show = np.concatenate((image_show, label_show, segmentation_show), axis=1)
            cv2.imshow("image - mask - prediction", cv2.cvtColor(full_show, cv2.COLOR_BGR2RGB))
            cv2.waitKey()

        print("epoch {}/{} ({}-th iteration)".format(e+1, num_epochs, i+1))
        print("loss=", loss.data.cpu().numpy())
        if (e % save_frequency) == 0:
            print("Model saved")
            torch.save(model.state_dict(), str(e) + ".torch")
        if (e+1) == num_epochs:
            print("Final model saved")
            torch.save(model.state_dict(), "final.torch")

def show():
    import cv2
    show_image = np.concatenate(
        (cv2.cvtColor(image[0].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB),
         cv2.cvtColor(label[0].numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)),
        axis=1
    )
    cv2.imshow("image", show_image)
    cv2.waitKey()
