import os

import torch
from torchvision import datasets
from torchvision.transforms import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
}


def get_dataloader(ds_dir, bs):
    train_datasets = datasets.ImageFolder(os.path.join(ds_dir, 'train'), data_transforms['train'])
    val_datasets = datasets.ImageFolder(os.path.join(ds_dir, 'val'), data_transforms['val'])
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=bs, shuffle=True, num_workers=4)
    val_dataloaders = torch.utils.data.DataLoader(val_datasets, batch_size=bs, shuffle=True, num_workers=4)

    print("=Classes: {}".format(train_datasets.classes))
    return train_dataloaders, val_dataloaders


# FIXME merge get_dataloader and this method
def get_testdataloader(ds_dir, bs, mode):
    test_datasets = datasets.ImageFolder(os.path.join(ds_dir, mode), data_transforms['val'])
    print("=Classes: {}".format(test_datasets.classes))
    return torch.utils.data.DataLoader(test_datasets, batch_size=bs, shuffle=False, num_workers=4)
