import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=0, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=0, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='SIDD')
        parser.add_argument('--pretrain_weights',type=str, default='.\\log\\UNet\\models\\model_best.pth', help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default ='UNet',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='denoising',  help='image restoration mode')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='.\\logs\\',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--train_dir', type=str, default ='../datasets/SIDD/train',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='../datasets/SIDD/val',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        return parser
