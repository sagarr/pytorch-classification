import argparse
import time

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import class_dataloader
from models import resnet


def validate(val_loader):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

    return (running_loss / len(val_loader)), (running_corrects.item() / len(val_loader))


def train(model, epochs, train_loader, val_loader, optimizer, criterion, reduce_lr_scheduler):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.item() / len(train_loader)

        val_loss, val_acc = validate(val_loader)
        reduce_lr_scheduler.step(val_loss)

        print('Train: Loss: {:.4f} Acc: {:.4f} '
              '| Val: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc, val_loss, val_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('=Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('=Best Val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a PyTorch model for classification.')
    parser.add_argument('-ds', '--dataset', help='', default='/tmp/dataset')
    parser.add_argument('-m', '--model_name', default="resnet18")
    parser.add_argument('-ep', '--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)

    args = parser.parse_args()

    if 'resnet' in args.model_name:
        model = resnet.get_model(4, args.model_name).cuda()
    else:
        raise AssertionError("unknown model")

    train_dataloaders, val_dataloaders = class_dataloader.get_dataloader(args.dataset, args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    reduce_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    model = train(model, args.epochs, train_dataloaders, val_dataloaders, optimizer, criterion, reduce_lr_scheduler)

    print("=Saving best model")
    model.save('models/' + args.model_name + ".pt")
