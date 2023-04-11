import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import argparse
import datetime
import torch
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from datasets.SARBuD_dataset import DARBuDDataset
from models import get_model
from losses.CBLLoss import CBLLoss
from losses.DualLoss import DualLoss
from utils.logger import get_logger
logger = None


# 全局变量
total_grad_out = []
total_grad_in = []
total_module = []



def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--ckpt-dir', type=str, default='/gemini/code/segmentation0329/checkpoint')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--log-dir', type=str, default="/gemini/code/segmentation0329/logs")
    parser.add_argument('--gpu-id', type=int, default=0, help='id of gpu to use ' '(only applicable to non-distributed training)')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model', type=str, default='unet_bc')
    parser.add_argument('--dataset', type=str, default='/gemini/code/dataset')

    return parser.parse_args()


def train(args, model, optimizer, bce_loss, cbl_loss, dual_loss, train_dataloader, val_dataloader, epoch, metric, mode='BC'):
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


            if mode == 'BE':
                output, cbl, bce = model(input)
                cblloss = 0.1 * (cbl_loss(cbl['cbl_1_8'], label_1_8)+ cbl_loss(cbl['cbl_1_4'], label_1_4) + cbl_loss(cbl['cbl_1_2'], label_1_2))  
                bceloss = 0.5 * (bce_loss(bce['bce_1_8'], label_1_8) + bce_loss(bce['bce_1_4'], label_1_4) + bce_loss(bce['bce_1_2'], label_1_2))
                total_loss = bce_loss(output, label) + cblloss + bceloss

                # total_loss = total_loss.mean()
                total_loss.backward()
                logger.info("Epoch[{}/{}] batch[{}] loss: {}".format(i, epoch, idx, total_loss))
                logger.info("Epoch[{}/{}] batch[{}] cblloss: {}".format(i, epoch, idx, cblloss))

            if mode == 'BC':
                output, pred = model(input)
                predloss = 0.5 * (dual_loss(pred['pred_1_8'], label_1_8) + dual_loss(pred['pred_1_4'], label_1_4) + dual_loss(pred['pred_1_2'], label_1_2))
                total_loss = bce_loss(output, label) + predloss

                # total_loss = total_loss.mean()
                total_loss.backward()
                logger.info("Epoch[{}/{}] batch[{}] loss: {}".format(i, epoch, idx, total_loss))
                logger.info("Epoch[{}/{}] batch[{}] predloss: {}".format(i, epoch, idx, predloss))
            


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
        logits, pred = model(input)
        total_loss = bce_loss(torch.sigmoid(logits), label) 

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
        transforms.Normalize(0.5, 0.5)
        # transforms.RandomCrop([256, 256])
    ])
    val_transform = transforms.Compose([
        transforms.Normalize([0], [255]),
        transforms.Normalize(0.5, 0.5)
    ])
    
    train_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/train/", train_transform), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/val/", val_transform), batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(DARBuDDataset(args.dataset + "/test/", val_transform), batch_size=args.batch_size, shuffle=False, drop_last=False)
    

    # init model
    # model = get_model(args.model).cuda()
    model = get_model(args.model).cuda()

    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # load ckpt
    if args.load_from:
        checkpoint = torch.load(args.load_from)
        model.load(checkpoint['model_state_dict'])

    # losses
    bce_loss = torch.nn.BCELoss()
    cbl_loss = CBLLoss([3,3], 1)
    dual_loss = DualLoss()

    # metrics
    metric = None
    train(args, model, optimizer, bce_loss, cbl_loss, dual_loss, train_dataloader, val_dataloader, args.epoch, metric)
    test(model,  bce_loss, cbl_loss, test_dataloader, metric)

if __name__ == '__main__':
    main()
