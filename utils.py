import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from models.cnn import *

from datasets import CIFAR10_truncated, ImageFolder_custom

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass


def load_cifar10_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=False, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=False, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)



def load_tinyimagenet_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])
    xray_train_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/train/', transform=transform)
    xray_test_ds = ImageFolder_custom(datadir+'tiny-imagenet-200/val/', transform=transform)

    X_train, y_train = np.array([s[0] for s in xray_train_ds.samples]), np.array([int(s[1]) for s in xray_train_ds.samples])
    X_test, y_test = np.array([s[0] for s in xray_test_ds.samples]), np.array([int(s[1]) for s in xray_test_ds.samples])

    return (X_train, y_train, X_test, y_test)


def record_net_data_stats(y_train, net_dataidx_map, logdir):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_clients, alpha=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'tinyimagenet':
        X_train, y_train, X_test, y_test = load_tinyimagenet_data(datadir)

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_clients)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_clients)}

    elif partition == "noniid":
        min_size = 0
        if alpha == 0.1:
            min_require_size = 64
        else:
            min_require_size = 10
        K = 10
        if dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def get_trainable_parameters(net, device='cpu'):
    'return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
    X = torch.empty(N, dtype=torch.float64, device=device)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel
    return X


def put_trainable_parameters(net, X):
    'replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel


def compute_accuracy(model, dataloader, get_confusion_matrix=False, args=None, device="cpu", multiloader=False):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0

    ce_loss = nn.CrossEntropyLoss()

    if "cuda" in device.type:
        ce_loss.cuda()

    loss_collector = []
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):

            images_1 = None

            if device != 'cpu':
                x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _, _, out = model(images_1 if torch.is_tensor(images_1) else x)

            loss_1 = 0
            if args.lambda_ce:
                loss_1 = ce_loss(out, target)
                _, pred_label = torch.max(out.data, 1)


            loss = loss_1 * args.lambda_ce 

            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix, avg_loss

    return correct / float(total), avg_loss


def get_size_label(dataset, num_classes):
    py = torch.zeros(num_classes)
    total = len(dataset.dataset)
    data_loader = iter(dataset)
    iter_num = len(data_loader)
    for it in range(iter_num):
        _, labels = next(data_loader)
        for i in range(num_classes):
            py[i] = py[i] + (i == labels).sum()
    py = py/(total)
    size_label = py*total
    return size_label


def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir + "trained_local_model" + str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return


def load_model(model, model_index, device="cpu"):
    with open("trained_local_model" + str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    if device == "cpu":
        model.to(device)
    else:
        model.cuda()
    return model


class ThresholdDescriptor:
    def __init__(self, name):
        super(ThresholdDescriptor, self).__init__()
        self.name = name
        if name in ["max", "mean", "min"]:
            self.degree = None
        elif name[-1].isdigit():
            self.degree = int(name[-1])

    def compute(self, value):
        if self.name == "mean":
            return torch.mean(value)
        elif self.name == "max":
            return torch.max(value)
        elif self.name == "min":
            return torch.min(value)
        elif self.degree and "max" in self.name:
            sorted_value, _ = torch.sort(value)
            return sorted_value[- self.degree]
        elif self.degree and "min" in self.name:
            sorted_value, _ = torch.sort(value)
            return sorted_value[self.degree]


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True,
                          transform=transform_train,
                          download=False)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=False)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'tinyimagenet':
        dl_obj = ImageFolder_custom
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_ds = dl_obj(datadir+'tiny-imagenet-200/train/', dataidxs=dataidxs, transform=transform_train)
        test_ds = dl_obj(datadir+'tiny-imagenet-200/val/', transform=transform_test)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)


    return train_dl, test_dl, train_ds, test_ds
