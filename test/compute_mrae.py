import glob
import os
import hdf5storage as hdf5
import numpy as np
import argparse


def compute_MRAE(gt, rec):
    gt_hyper = gt
    rec_hyper = rec
    error = np.abs(rec_hyper - gt_hyper) / gt_hyper
    mrae = np.mean(error.reshape(-1))
    return mrae

def compute_RMSE(gt, rec):
    error = np.power(gt - rec, 2)
    rmse = np.sqrt(np.mean(error))
    return rmse

def main():
    path_rec = opt.path_rec
    path_gt = opt.path_gt

    name_rec_list = glob.glob(os.path.join(path_rec, '*.mat'))
    name_gt_list = glob.glob(os.path.join(path_gt, '*.mat'))
    name_rec_list.sort()
    name_gt_list.sort()

    mrae_all = []
    rmse_all = []

    for i in range(len(name_gt_list)):
        hyper_rec = hdf5.loadmat(name_rec_list[i])['cube']
        hyper_gt = hdf5.loadmat(name_gt_list[i])['cube']
        if hyper_gt.min()<= 0.:
            print(os.path.basename(name_gt_list[i]), end=' ')
            print('This file is not suitable for compute the MRAE indicator.')
            continue
        hyper_rec = np.clip(hyper_rec, 0,1)
        mrae = compute_MRAE(hyper_gt, hyper_rec)
        rmse = compute_RMSE(hyper_gt, hyper_rec)
        print(os.path.basename(name_gt_list[i]), end=' ')
        print('mrae: '+str(mrae)+',  rmse: '+str(rmse))
        mrae_all.append(mrae)
        rmse_all.append(rmse)
    print('The average mrae is: '+str(sum(mrae_all)/len(mrae_all)))
    print('The average rmse is: '+str(sum(rmse_all)/len(rmse_all)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSR_test")
    parser.add_argument("--path_rec", type=str, default='./test_results', help="The path of the reconstructed valid spectral data.")
    parser.add_argument("--path_gt", type=str, default='../Dataset/Valid_spectral', help="The path of the ground truth valid spectral data.")
    opt = parser.parse_args()
    main()