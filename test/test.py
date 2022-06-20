import torch
import os
import numpy as np
import cv2
from DRCR import DRCR
import glob
import hdf5storage as hdf5
import time
import argparse


def get_reconstruction_gpu(input, model):
    """As the limited GPU memory split the input."""
    model.eval()
    var_input = input.cuda()
    with torch.no_grad():
        start_time = time.time()
        var_output1 = model(var_input[:,:,:-2,:])
        var_output2 = model(var_input[:,:,2:,:])
        var_output = torch.cat([var_output1, var_output2[:,:,-2:,:]], 2)
        end_time = time.time()

    return end_time-start_time, var_output.cpu()

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="SSR_test")
parser.add_argument("--RGB_dir", type=str, default='../Dataset/Valid_RGB', help="absolute Input_RGB_path")
parser.add_argument("--model_dir", type=str, default='./model/model.pth', help="absolute Model_path")
parser.add_argument("--result_dir", type=str, default='./test_results', help="absolute Save_Result_path")
opt = parser.parse_args()

img_path = opt.RGB_dir
model_path = opt.model_dir
result_path = opt.result_dir

var_name = 'cube'
# save results
if not os.path.exists(result_path):
    os.makedirs(result_path)
model = DRCR(3, 31, 100, 10)
save_point = torch.load(model_path)
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)
model = model.cuda()

img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
img_path_name.sort()

for i in range(len(img_path_name)):
    # load rgb images
    print(img_path_name[i].split('/')[-1])
    rgb = cv2.imread(img_path_name[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb)
    rgb = rgb / rgb.max()
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    rgb = torch.from_numpy(rgb).float()
    use_time, temp_hyper = get_reconstruction_gpu(rgb, model)
    img_res = temp_hyper.numpy() * 1.0
    img_res = np.transpose(np.squeeze(img_res), [1, 2, 0])
    img_res_limits = np.minimum(img_res, 1.0)
    img_res_limits = np.maximum(img_res_limits, 0)

    mat_name = img_path_name[i].split('/')[-1][:-4] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)
    hdf5.savemat(mat_dir, {var_name: img_res}, format='7.3', store_python_metadata=True)