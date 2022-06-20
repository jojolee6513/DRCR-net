import os
import cv2
import glob
import numpy as np
import argparse
import hdf5storage as hdf5
import tqdm

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='../Dataset', help="data path")
parser.add_argument("--patch_size", type=int, default=128, help="data patch size")
parser.add_argument("--stride", type=int, default=64, help="data patch stride")
parser.add_argument("--train_data_path", type=str, default='./dataset/Train', help="preprocess_data_path")
opt = parser.parse_args()


def main():
    if not os.path.exists(opt.train_data_path):
        os.makedirs(opt.train_data_path)

    process_data(patch_size=opt.patch_size, stride=opt.stride, mode='train')


def normalize(data, max_val, min_val):
    return (data-min_val)/(np.float32(max_val-min_val))


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def process_data(patch_size, stride, mode):
    if mode == 'train':
        print("\nprocess training set ...\n")
        patch_num = 1
        filenames_hyper = glob.glob(os.path.join(opt.data_path, 'Train_spectral', '*.mat'))
        filenames_rgb = glob.glob(os.path.join(opt.data_path, 'Train_RGB', '*.jpg'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        print(len(filenames_rgb))
        # for k in range(1):  # make small dataset
        for k in tqdm.tqdm(range(len(filenames_rgb))):
            print([filenames_rgb[k][-16:]])

            mat = hdf5.loadmat(filenames_hyper[k])
            hyper = np.float32(np.array(mat['cube']))
            hyper = np.transpose(hyper, [2, 0, 1])
            if hyper.min() <= 0:
                print('This file contains non-positive values and is not suitable for Training!')
                continue
            hyper = normalize(hyper, max_val=1., min_val=0.)
            # load rgb image
            rgb = cv2.imread(filenames_rgb[k])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, [2, 0, 1])
            rgb = normalize(np.float32(rgb), max_val=rgb.max(), min_val=0.)
            # creat patches
            patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
            patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
            for j in range(patches_rgb.shape[3]):
                print("generate training sample #%d" % patch_num)
                sub_hyper = patches_hyper[:, :, :, j]
                sub_rgb = patches_rgb[:, :, :, j]
                train_data_path = os.path.join(opt.train_data_path, 'train'+str(patch_num)+'.mat')
                print(train_data_path)
                hdf5.savemat(train_data_path, {'rad': sub_hyper}, format='7.3')
                hdf5.savemat(train_data_path, {'rgb': sub_rgb}, format='7.3')
                patch_num += 1

        print("\ntraining set: # samples %d\n" % (patch_num-1))


if __name__ == '__main__':
    main()

