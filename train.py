import os
import argparse
import datetime
import torch
from torchvision import transforms
# from torchsummary import summary
import matplotlib.pyplot as plt
from datasets.SARBuD_dataset import DARBuDDataset
from models import get_model
from losses.CBLLoss import CBLLoss
from utils.logger import get_logger
logger = None



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--ckpt-dir', type=str, default='/gemini/code/segmentation0329/checkpoint')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--log-dir', type=str, default="/gemini/code/segmentation0329/logs")
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ' '(only applicable to non-distributed training)')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--dataset', type=str, default='/gemini/code/dataset')

    return parser.parse_args()


def train(args, model, optimizer, bce_loss, cbl_loss, train_dataloader, val_dataloader, epoch, metric):
    optimizer.zero_grad()
    acc = 0
    for i in range(epoch):
        logger.info("Start Training epoch[{}/{}]".format(i, epoch))
        logger.info("==========================================================")
        model.train()
        for idx, (input, label, label_1_2, label_1_4, label_1_8) in enumerate(train_dataloader):
            optimizer.zero_grad()
            input = input.cuda()
            label = label.cuda()
            label_1_2 = label_1_2.cuda()
            label_1_4 = label_1_4.cuda()
            label_1_8 = label_1_8.cuda()
            logits, cbl_1_8, bce_1_8, cbl_1_4, bce_1_4, cbl_1_2, bce_1_2 = model(input)
            
            total_loss = \
                bce_loss(torch.sigmoid(logits), label) + \
                0.1*cbl_loss(cbl_1_8, label_1_8) + \
                0.1*cbl_loss(cbl_1_4, label_1_4) + \
                0.1*cbl_loss(cbl_1_2, label_1_2) + \
                0.5*bce_loss(torch.sigmoid(bce_1_8), label_1_8) + \
                0.5*bce_loss(torch.sigmoid(bce_1_4), label_1_4) + \
                0.5*bce_loss(torch.sigmoid(bce_1_2), label_1_2) 
            total_loss = total_loss.mean()

            cblloss = cbl_loss(cbl_1_2, label_1_2)

            logger.info("Epoch[{}/{}] batch[{}] loss: {}".format(i, epoch, idx, total_loss))
            logger.info("Epoch[{}/{}] batch[{}] cblloss: {}".format(i, epoch, idx, cblloss))
            total_loss.backward()
            optimizer.step()

        new_acc = test(model, bce_loss, cbl_loss, val_dataloader, metric)
        if new_acc > acc:
            torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, "{}/{}-{}-{}.pt".format(args.ckpt_dir, args.model, i, new_acc))
            logger.info("Best acc ever: {}, before is {}".format(new_acc, acc))
            acc = new_acc
        logger.info("==========================================================")

def test(model, bce_loss, cbl_loss, dataloadder, metric):
    model.eval()
    for idx, (input, label, label_1_2, label_1_4, label_1_8) in enumerate(dataloadder):
        input = input.cuda()
        label = label.cuda()
        label_1_2 = label_1_2.cuda()
        label_1_4 = label_1_4.cuda()
        label_1_8 = label_1_8.cuda()
        logits, cbl_1_8, bce_1_8, cbl_1_4, bce_1_4, cbl_1_2, bce_1_2 = model(input)
        total_loss = \
            bce_loss(torch.sigmoid(logits), label) + \
            0.1*cbl_loss(cbl_1_8, label_1_8) + \
            0.1*cbl_loss(cbl_1_4, label_1_4) + \
            0.1*cbl_loss(cbl_1_2, label_1_2) + \
            0.5*bce_loss(torch.sigmoid(bce_1_8), label_1_8) + \
            0.5*bce_loss(torch.sigmoid(bce_1_4), label_1_4) + \
            0.5*bce_loss(torch.sigmoid(bce_1_2), label_1_2) 
    if metric:
        logger.info("Testing acc: {}".format(metric))
    return 0.1


def main():
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    global logger 
    logger = get_logger(args.log_dir + "/{}-{}-{}-{}.csv".format(
        args.model, 
        args.epoch, 
        args.batch_size, 
        datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        ))
    logger.info(args)
    # set cudnn_benchmark
    if args.deterministic:
        torch.backends.cudnn.benchmark = True
    
    # init dataloader
    train_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Normalize([0], [255]),
        transforms.Normalize(0.5, 0.5),
        # transforms.RandomCrop([256, 256])
    ])
    val_transform = transforms.Compose([
        transforms.Normalize([0, 0, 0, 0], [255, 255, 255, 255])])
    
    train_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/train/", train_transform), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/val/", val_transform), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/test/", val_transform), batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # init model
    model = get_model(args.model).cuda()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # load ckpt
    if args.load_from:
        checkpoint = torch.load(args.load_from)
        model.load(checkpoint['model_state_dict'])

    # losses
    bce_loss = torch.nn.BCELoss()
    cbl_loss = CBLLoss([3,3], 10)

    # metrics
    metric = None
    train(args, model, optimizer, bce_loss, cbl_loss, train_dataloader, val_dataloader, args.epoch, metric)
    test(model,  bce_loss, cbl_loss, test_dataloader, metric)

if __name__ == '__main__':
    main()
