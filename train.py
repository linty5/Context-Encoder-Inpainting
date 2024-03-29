import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'pre-train ot not')
    parser.add_argument('--save_mode', type = str, default = 'iter', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 5, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--save_name_mode', type = bool, default = True, help = 'True for concise name, and False for exhaustive name')
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 40, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0001, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0001, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'weight decay for optimizer')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 200000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 1, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--lambda_rec', type = float, default = 0.01, help = 'coefficient for Reconstruction Loss')
    parser.add_argument('--lambda_adv', type = float, default = 10, help = 'coefficient for wGAN Loss')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 1, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 1, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--bottleneck_channels', type = int, default = 4000, help='bottle neck channels between Encoder and Decoder')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'LSGAN', help = 'type of GAN: [LSGAN | WGAN], LSGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Dataset parameters
    parser.add_argument('--baseroot_A', type = str, default = '/media/ztt/6864FEA364FE72E4/zhaoyuzhi/ILSVRC2012_train256', help = 'domain A baseroot')
    parser.add_argument('--baseroot_B', type = str, default = '/media/ztt/6864FEA364FE72E4/zhaoyuzhi/old image processed/Grayscale/photo', help = 'domain B baseroot')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    #opt.gpu_ids = -1

    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    
    # ----------------------------------------
    #                 Inpainting
    # ----------------------------------------
    trainer.Inpainting(opt)
