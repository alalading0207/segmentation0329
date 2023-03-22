import argparse

import torch
from torchvision import transforms
from datasets.SARBuD_dataset import DARBuDDataset
from models import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--epoch', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--dataset', type=str, default='./')

    return parser.parse_args()

def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, epoch, metric):
    optimizer.zero_grad()
    for i in range(epoch):
        print("Start Training epoch[{}/{}]".format(i, epoch))
        print("==========================================================")
        model.train()
        for input, label in train_dataloader:
            optimizer.zero_grad()
            input = input.cuda()
            label = label.cuda()
            output = model(input)
            loss = loss_fn(label, output)
            print("Training epoch[{}/{}] loss: {}".format(i, epoch, loss))
            loss.backward()
            optimizer.step()
        test(model, val_dataloader, loss_fn, metric)
        print("==========================================================")

def test(model, dataloadder, loss_fn, metric):
    model.val()
    for input, label in dataloadder:
        input = input.cuda()
        label = label.cuda()
        output = model(input)
        loss = loss_fn(label, output)
    if metric:
        print("Testing acc: {}".format(metric))


def main():
    args = parse_args()

    # set cudnn_benchmark
    if args.deterministic:
        torch.backends.cudnn.benchmark = True
    
    # init dataloader
    train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Normalize([0, 0, 0, 0], [255, 255, 255, 255]),
        # transforms.RandomCrop([256, 256])
    ])
    val_transform = transforms.Compose([
        transforms.Normalize([0, 0, 0, 0], [255, 255, 255, 255])])
    train_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/train/", train_transform), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/val/", val_transform), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/test/", val_transform), batch_size=args.batch_size, shuffle=False, drop_last=False)
    # init model
    model = get_model(args.model).cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # loss 
    loss_fn = torch.nn.MSELoss()
    # metrics
    metric = None
    train(model, optimizer, loss_fn, train_dataloader, val_dataloader, args.epoch, metric)
    test(model, test_dataloader, loss_fn, metric)

if __name__ == '__main__':
    main()
