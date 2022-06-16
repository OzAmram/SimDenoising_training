import numpy as np

import sys
sys.path.append(".local/lib/python{}.{}/site-packages".format(sys.version_info.major,sys.version_info.minor))

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from models import DnCNN, DnPointCloudCNN, PatchLoss, WeightedPatchLoss
import uproot as up
import numpy as np
import torch.utils.data as udata
import dataset as dat
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
import random
import time
from torchinfo import summary

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model(trained_model, device, dataset, args):
    if "jit" in trained_model:
        return torch.jit.load(trained_model, map_location=device)
    else:
        if(args.imageOnly):
            #model = DnCNN(channels=1, num_of_layers=9, kernel_size=3, features=100).to(device)
            model = DnCNN(channels=dataset.nfeatures, num_of_layers=args.num_layers, kernel_size=args.kernelSize, features=args.features).to(device=device)
        else:
            model = DnPointCloudCNN(channels=dataset.nfeatures, num_init_CNN_layers=args.num_layers//2, num_post_CNN_layers = args.num_layers//2, 
                kernel_size=args.kernelSize, features=args.features, set_feature_size = args.numSetFeats).to(device)

        model.load_state_dict(torch.load(trained_model, map_location=device))
        model.eval()
        torch.no_grad()
        return model

def main():
    parser = ArgumentParser(description="DnCNN", config_options=MagiConfigOptions(), formatter_class=ArgumentDefaultsRawHelpFormatter)

    parser.add_argument("--model", type=str, required=True, help='Path to .pth file with saved model')
    parser.add_argument("--imageOnly", default = False, action = 'store_true', help = "Use input image only")
    parser.add_argument("--extraImages", default = False, action = 'store_true', help = "Also use images not based on energy")
    parser.add_argument("--applyAugs", default = True, help = "Apply augmentations (flips and rotations) to images")
    parser.add_argument("--numSetFeats", type = int, default = 4, help = "Number of features per particle")
    parser.add_argument("--numpy", type=str, default="test.npz", help='Name of .npz file with CNN-enhanced low quality (fuzzy) data')
    parser.add_argument("--fileSharp", "--testfileSharp", dest = 'testfileSharp', type=str, default=[], nargs='+', help='Path to higher quality .root file for making plots')
    parser.add_argument("--fileFuzz", "--testfileFuzz", dest = 'testfileFuzz', type=str, default=[], nargs='+', help='Path to lower quality .root file for making plots')
    parser.add_argument("--randomseed", type=int, default=0, help="Initial value for random.seed()")
    parser.add_argument("--transform", type=str, default=[], nargs='*', choices=dat.RootDataset.allowed_transforms, help="transform(s) for input data")
    parser.add_argument("--batchSize", type=int, default=100, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loaders")
    parser.add_argument("--num-layers", type=int, default=9, help="Number of total layers in the CNN")
    parser.add_argument("--kernelSize", type=int, default=3, help="Size of kernel in CNN")
    parser.add_argument("--features", type=int, default=9, help="Number of features in CNN layers")
    parser.add_argument("--verbose", default=False, action="store_true", help="enable verbose printouts")
    args = parser.parse_args()

    # backward compatibility
    if not isinstance(args.transform,list): args.transform = [args.transform]

    # choose cpu or gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    random.seed(args.randomseed)
    torch.manual_seed(args.randomseed)
    #dataset = dat.RootDataset(args.testfileFuzz,args.testfileSharp,args.transform,output=True, applyAugs = args.applyAugs, imageOnly = args.imageOnly)

    dataset = dat.RootDataset(sharp_root=args.testfileSharp, fuzzy_root=args.testfileFuzz, transform=args.transform, 
            applyAugs = args.applyAugs, imageOnly = args.imageOnly, numSetFeats = args.numSetFeats, extraImages = args.extraImages, output=True)
    loader = udata.DataLoader(dataset=dataset, batch_size=args.batchSize, num_workers=args.num_workers)

    model = load_model(args.model, device, dataset, args)
    #summary(model, [(1,3,50,50)])

    outputs = []
    inference_time = 0
    total_time = 0
    for i, data in enumerate(loader):
        t1 = time.time()
        if(args.imageOnly):
            _, fuzzy, means, stdevs = data
            fuzzy = fuzzy.float().to(device)
            t2 = time.time()
            output = model(fuzzy)
        else:
            (_, fuzzy, means, stdevs), feats = data
            fuzzy = fuzzy.float().to(device)
            feats = feats.float().to(device)
            t2 = time.time()
            output = model(fuzzy, feats)

        t3 = time.time()
        output = output.squeeze(1).cpu().detach().numpy()
        output = dataset.unnormalize(output,means=means,stdevs=stdevs)
        if i==0: outputs = output
        else: outputs = np.concatenate((outputs,output))
        del _
        del fuzzy
        del output
        t4 = time.time()
        inference_time += t3-t2
        total_time += t4-t1
    np.savez(args.numpy, outputs)
    print("Outputing to %s" % args.numpy)
    if args.verbose: print("Nevents: {}\nInference time: {} s\nTotal time: {} s".format(len(outputs),inference_time, total_time))

if __name__ == "__main__":
    main()
