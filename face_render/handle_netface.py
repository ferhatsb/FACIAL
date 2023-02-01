import os, sys
import numpy as np
from scipy.io import loadmat, savemat
import glob
from scipy.signal import savgol_filter
import argparse

parser = argparse.ArgumentParser(description='netface_setting')
parser.add_argument('--param_folder', type=str, default='../video_preprocess/train1_deep3Dface')

opt = parser.parse_args()

param_folder = opt.param_folder

mat_path_list = sorted(glob.glob(os.path.join(param_folder, '*.mat')))
len_mat = len(mat_path_list)
faceshape = np.zeros((len_mat, 257), float)

for i in range(1, len_mat + 1):
    mat = loadmat(os.path.join(param_folder, str(i).zfill(6) + '.mat'))
    faceshape[i - 1, :] = np.hstack((mat['id'], mat['exp'], mat['tex'], mat['angle'], mat['gamma'], mat['trans']))

frames_out_path = os.path.join(param_folder, 'train1.npz')
np.savez(frames_out_path, face=faceshape)
