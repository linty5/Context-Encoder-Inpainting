import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

def Inpainting(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize Generator
    G_I = utils.create_generator(opt)
    D_I = utils.create_discriminator(opt)


    # To device
    if opt.multi_gpu:
        G_I = nn.DataParallel(G_I)
        G_I = G_I.cuda()
        D_I = nn.DataParallel(D_I)
        D_I = D_I.cuda()
    else:
        G_I = G_I.cuda()
        G_I = G_I.cuda()
        D_I = D_I.cuda()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(G_I.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.RMSprop(D_I.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, G_I):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(G_I.module, 'G_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(G_I.module, 'G_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(G_I, 'G_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(G_I, 'G_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DomainTransferDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, img, valid in enumerate(dataloader):

            # To device
            img = img.cuda()
            valid = valid.cuda()

            # Adversarial ground truth
            valid = Tensor(np.ones((img.shape[0], 1, 16, 16)))
            fake = Tensor(np.zeros((img.shape[0], 1, 16, 16)))

            # Train Generator
            optimizer_G.zero_grad()

            # GAN Loss
            fake = G_I(img)
            loss_GAN = criterion_MSE(D_I(fake), valid)

            # wGAN Loss
            fake = G_I(img)
            loss_wGAN = -torch.mean(D_I(valid)) + torch.mean(D_I(fake))

            # Reconstruction Loss(L2)
            loss_rec = criterion_MSE(fake, valid)

            # Overall Loss and optimize
            loss = opt.lambda_rec * loss_rec + opt.lambda_adv * loss_wGAN
            loss.backward()
            optimizer_G.step()

            for p in D_I.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train Discriminator
            optimizer_D.zero_grad()
    
            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [G wGAN Loss: %.4f] [G rec Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_wGAN.item(), loss_rec.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), G_I)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)
