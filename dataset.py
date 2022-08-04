import sys
import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import torch.utils.data as udata

def get_flips():
    flipx = random.randint(0, 1)
    flipy = random.randint(0, 1)
    rot = random.randint(0, 3)
    return flipx, flipy, rot

def get_tree(file_path):
    file = up.open(file_path)
    tree = file["g4SimHits/tree"]
    return tree

def get_branch(file_paths, branch_name = 'bin_weights'):
    branch = []
    for i, elem in enumerate(file_paths):
        tree = get_tree(file_paths[i])
        events = tree[branch_name].array()
        branch = np.concatenate((branch, events))
    branch = np.asarray(branch)
    return branch

def get_feature_branches(file_paths, keys = None, num_to_keep = 100):
    if(keys is None): keys = ["step_E", "step_x", "step_y", "step_t"]
    keys.remove("step_E")
    E_output = None
    f_output = None

    for i, elem in enumerate(file_paths):
        tree = get_tree(file_paths[i])
        E = tree["step_E"].array()
        arrs = [tree[key].array() for key in keys]

        E_outs = np.zeros((len(E), num_to_keep))
        outs = np.zeros((len(keys), len(E), num_to_keep))

        #arrays are not uniform size, so need to loop through
        for j in range(len(E)):
            #normalize E (?)

            #Get indices of largest energy steps
            np_E = np.array(E[j])
            np_arr = np.array([ a[j] for a in arrs])
            top_idxs = np.argpartition(np_E, kth = -num_to_keep)[-num_to_keep:]

            E_outs[j] = np_E[top_idxs]
            outs[:,j,:] = np_arr[:, top_idxs]

        if(i == 0):
            E_output = E_outs
            f_output = outs
            
        else:
            E_output = np.concatenate((E_output, E_outs), axis = 0)
            f_output = np.concatenate((f_output, outs), axis = 1)

    #Group features per event together by reorganizing array
    f_output = f_output.swapaxes(0,2).swapaxes(0,1)
    return (E_output, f_output)


class RootDataset(udata.Dataset):
    allowed_transforms = ["none","normalize","normalizeSharp","log10","sqrt"]
    nfeatures = 1
    def __init__(self, fuzzy_root, sharp_root, transform=[], shuffle=True, output=False, applyAugs = True, imageOnly=True,
            numSetFeats = 4, extraImages = False):
        print(fuzzy_root)
        print(sharp_root)
        self.imageOnly = imageOnly
        self.applyAugs = applyAugs
        # assume bin configuration is the same for all files
        sharp_tree = get_tree(sharp_root[0])
        self.xbins = sharp_tree["xbins"].array().to_numpy()[0]
        self.ybins = sharp_tree["ybins"].array().to_numpy()[0]
        self.xmin = sharp_tree["xmin"].array().to_numpy()[0]
        self.ymin = sharp_tree["ymin"].array().to_numpy()[0]
        self.xmax = sharp_tree["xmax"].array().to_numpy()[0]
        self.ymax = sharp_tree["ymax"].array().to_numpy()[0]

        # other member variables
        self.transform = transform
        self.means = None
        self.stdevs = None
        self.do_unnormalize = False
        self.output = output
        unknown_transforms = [transform for transform in self.transform if transform not in self.allowed_transforms]
        if len(unknown_transforms)>0:
            raise ValueError("Unknown transforms: {}".format(unknown_transforms))

        # get data in np format
        self.sharp_branch = get_branch(sharp_root)
        self.fuzzy_branch_E = get_branch(fuzzy_root)
        # reshape to image tensor
        self.sharp_branch = self.sharp_branch.reshape((self.sharp_branch.shape[0],1,self.xbins,self.ybins))
        self.fuzzy_branch_E = self.fuzzy_branch_E.reshape((self.fuzzy_branch_E.shape[0],1,self.xbins,self.ybins))
        if(extraImages):
            self.fuzzy_branch_time = get_branch(fuzzy_root, 't_Eavg_bin_weights').reshape(self.fuzzy_branch_E.shape[0], 1, self.xbins, self.ybins)
            #self.fuzzy_branch_time = get_branch(fuzzy_root, 't_avg_bin_weights').reshape(self.fuzzy_branch_E.shape[0], 1, self.xbins, self.ybins)
            #self.fuzzy_branch_time = get_branch(fuzzy_root, 't_max_bin_weights').reshape(self.fuzzy_branch_E.shape[0], 1, self.xbins, self.ybins)
            self.fuzzy_branch_n = get_branch(fuzzy_root, 'n_bin_weights').reshape(self.fuzzy_branch_E.shape[0], 1, self.xbins, self.ybins)
            self.nfeatures = 3
        # apply transforms if any (in order)
        for transform in self.transform:
            if transform=="log10":
                self.sharp_branch = np.log10(self.sharp_branch+1.0)
                self.fuzzy_branch_E = np.log10(self.fuzzy_branch_E+1.0)
            elif transform=="sqrt":
                self.sharp_branch = np.sqrt(self.sharp_branch)
                self.fuzzy_branch_E = np.sqrt(self.fuzzy_branch_E)
            elif transform.startswith("normalize"):
                norm_branch = self.sharp_branch if transform=="normalizeSharp" else self.fuzzy_branch_E
                self.means = np.average(norm_branch, axis=(1,2,3))[:,None,None,None]
                self.stdevs = np.std(norm_branch, axis=(1,2,3))[:,None,None,None]
                self.sharp_branch = np.divide(self.sharp_branch-self.means,self.stdevs,where=self.stdevs!=0)
                self.fuzzy_branch_E = np.divide(self.fuzzy_branch_E-self.means,self.stdevs,where=self.stdevs!=0)
                if(extraImages):

                    self.means_time = np.average(self.fuzzy_branch_time, axis=(1,2,3))[:,None,None,None]
                    self.stdevs_time = np.std(self.fuzzy_branch_time, axis=(1,2,3))[:,None,None,None]
                    self.means_n = np.average(self.fuzzy_branch_n, axis=(1,2,3))[:,None,None,None]
                    self.stdevs_n = np.std(self.fuzzy_branch_n, axis=(1,2,3))[:,None,None,None]
                    self.fuzzy_branch_time = np.divide(self.fuzzy_branch_time-self.means_time,self.stdevs_time,where=self.stdevs_time!=0)
                    self.fuzzy_branch_n = np.divide(self.fuzzy_branch_n-self.means_n,self.stdevs_n,where=self.stdevs_n!=0)


        self.feats = None
        if(not self.imageOnly):
            num_to_keep = 100
            keys_tot = ["step_E", "step_x", "step_y", "step_t", "step_z", "step_length", "delta_px", "delta_py", "delta_pz", "delta_t"]
            keys = keys_tot[:numSetFeats]
            step_E, other = get_feature_branches(fuzzy_root, num_to_keep = num_to_keep, keys = keys)
            print("step_E", step_E.shape)
            print("other", other.shape)
            #center x and y
            #step_y -= (self.ymax - self.ymin)/2
            #step_y /= (self.ymax - self.ymin)/2

            #step_x -= (self.xmax - self.xmin)/2
            #step_x /= (self.xmax - self.xmin)/2
            
            step_E = np.sqrt(step_E)

            #if(numSetFeats == 4):
            #    self.feats = np.stack((step_x, step_y, step_E, step_t), axis = 2)
            #else:
            #    self.feats = np.stack((step_x, step_y, step_E), axis = 2)
            step_E = np.expand_dims(step_E, axis = -1)
            print(other.shape)
            print(step_E.shape)
            self.feats = np.concatenate((other, step_E), axis = 2)
            print(self.feats.shape)
            means = np.average(self.feats, axis = 0)
            stds = np.std(self.feats, axis = 0)
            print(self.feats.shape, means.shape, stds.shape)
            self.feats = (self.feats - means)/ stds


        if(extraImages):
            self.fuzzy_branch = np.concatenate((self.fuzzy_branch_E, self.fuzzy_branch_time, self.fuzzy_branch_n), axis =1)
        else:
            self.fuzzy_branch  = self.fuzzy_branch_E



        # apply random rotation/flips consistently for both datasets
        if(applyAugs):
            for idx in range(self.sharp_branch.shape[0]):
                flipx, flipy, rot = get_flips()
                def do_flips(branch,idx,flipx,flipy,rot):
                    if flipx: branch[idx] = np.fliplr(branch[idx])
                    if flipy: branch[idx] = np.flipud(branch[idx])
                    if rot: branch[idx] = np.rot90(branch[idx], rot, (1,2))
                do_flips(self.sharp_branch,idx,flipx,flipy,rot)
                do_flips(self.fuzzy_branch,idx,flipx,flipy,rot)
        # fix shapes
        #print(self.fuzzy_branch.shape)
        #self.sharp_branch = self.sharp_branch.squeeze(1)
        #self.fuzzy_branch = self.fuzzy_branch.squeeze(1)
        if self.means is not None:
            self.means = np.squeeze(self.means,1)
        if self.stdevs is not None:
            self.stdevs = np.squeeze(self.stdevs,1)

    def __len__(self):
        if len(self.sharp_branch) == len(self.fuzzy_branch):
            return len(self.sharp_branch)
        else:
            raise RuntimeError("Sharp and fuzzy dataset lengths do not match")

    def __getitem__(self, idx):
        if self.do_unnormalize:
            imgs = (self.unnormalize(self.sharp_branch[idx],idx=idx).squeeze(),
                   self.unnormalize(self.fuzzy_branch[idx],idx=idx).squeeze())
        else:
            if self.output and any(transform.startswith("normalize") for transform in self.transform):
                imgs = (self.sharp_branch[idx], self.fuzzy_branch[idx], self.means[idx], self.stdevs[idx])
            else:
                imgs = (self.sharp_branch[idx], self.fuzzy_branch[idx])
        if(self.imageOnly): return imgs
        else: return imgs, self.feats[idx]

    # assumes array is same size as inputs
    def unnormalize(self,array,idx=None,means=None,stdevs=None):
        if means is None: means = self.means
        else: means = np.asarray(means)
        if stdevs is None: stdevs = self.stdevs
        else: stdevs = np.asarray(stdevs)

        # unapply transform(s) in reverse order
        for transform in reversed(self.transform):
            if transform=="log10":
                array = np.power(10,array)-1.0
            elif transform=="sqrt":
                array = np.power(array,2)
            elif transform.startswith("normalize"):
                if idx==None:
                    array = array*stdevs+means
                else:
                    array = array*stdevs[idx].squeeze()+means[idx].squeeze()
        return array

if __name__=="__main__":
    random.seed(0)
    dataset = RootDataset([sys.argv[1]], [sys.argv[2]])
    truth, noise = dataset.__getitem__(0)
    print("Default:")
    print("truth:",truth,truth.shape)
    print("noisy:",noise,noise.shape)
    torch.manual_seed(0)
    dataset = RootDataset([sys.argv[1]], [sys.argv[2]], ["log10","normalize"])
    truth, noise = dataset[0]
    print("Normalized:")
    print("truth:",truth,truth.shape)
    print("noisy:",noise,noise.shape)
    print("means:",dataset.means[0],dataset.means.shape)
    print("stdevs:",dataset.stdevs[0],dataset.stdevs.shape)
    dataset.do_unnormalize = True
    truth, noise = dataset.__getitem__(0)
    print("Unnormalized:")
    print("truth:",truth,truth.shape)
    print("noisy:",noise,noise.shape)
