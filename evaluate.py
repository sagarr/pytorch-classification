import operator
from functools import reduce

import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from class_dataloader import get_testdataloader
from models import resnet


def get_model(model_name, wt_file):
    if 'resnet' in model_name:
        model = resnet.get_model(4, model_name).cuda()
    else:
        raise AssertionError("unknown model")

    state_dict = torch.load(wt_file)
    model.load_state_dict(state_dict)
    return model


def evaluate(model, loader):
    model.eval()

    acc = 0
    ps = []
    lbls = []
    confusion_matrix = torch.zeros(4, 4)
    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        lbls.append(labels.data.cpu().numpy().tolist())
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            ps.append(preds.data.cpu().numpy().tolist())
            acc += torch.sum(preds == labels.data)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return acc, confusion_matrix, ps, lbls,


if __name__ == '__main__':
    name = 'resnet152'
    wt_file = ''
    model = get_model(name, wt_file)

    bs = 4
    ds_dir = ''
    test_loader = get_testdataloader(ds_dir, 4, 'test')

    acc, cf, preds, lbls = evaluate(model, test_loader)

    print("=Accuracy per class:")
    print(cf.diag() / cf.sum(1))

    preds = reduce(operator.concat, preds)
    lbls = reduce(operator.concat, lbls)
    print("=Confusion matrix:")
    print(confusion_matrix(lbls, preds))

    acc = acc.item() / (len(test_loader) * bs)
    print()
    print('=Eval: Acc: {:.4f}'.format(acc))
