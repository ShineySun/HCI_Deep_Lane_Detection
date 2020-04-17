import sys
# get arg value
import opts
import math
import importlib
from preprocess import *
import _init_paths
import torch


def main():
    # Cuda Setting Value : Default = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # opts.py : get arg parser
    opt = opts.parse()

    # Cuda Current Device
    print(("device id: {}".format(torch.cuda.current_device())))
    # Pytorch version : 0.4.1
    print("torch.version",torch.__version__)
    # Cuda version : 9.0.1
    print("cuda_version",torch.version.cuda)

    



if __name__ == '__main__':
    main()
