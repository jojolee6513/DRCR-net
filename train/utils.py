from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import hdf5storage


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
            'epoch': epoch,
            'iter': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }
    
    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()    
    loss_csv.close


class Loss_train(nn.Module):
    def __init__(self):
        super(Loss_train, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        # error = torch.abs(outputs - label)
        mrae = torch.mean(error.view(-1))
        return mrae


class Loss_valid(nn.Module):
    def __init__(self):
        super(Loss_valid, self).__init__()

    def forward(self, outputs, label):
        error = torch.abs(outputs - label) / label
        # error = torch.abs(outputs - label)
        mrae = torch.mean(error.view(-1))
        return mrae



