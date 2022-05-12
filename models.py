"""
Adapted from https://github.com/SaoYan/DnCNN-PyTorch/blob/master/models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable


class DnPointCloudCNN(nn.Module):
    def __init__(self, channels=1, img_size = 100, num_init_CNN_layers=3, num_post_CNN_layers = 3, kernel_size=3, features=100, set_feature_size = 3, set_latent_size = 50):
        super(DnPointCloudCNN, self).__init__()
        self.set_feature_size = set_feature_size
        self.set_latent_size = set_latent_size
        self.img_size = img_size

        self.set_feature_extractor = nn.Sequential(
            nn.Linear(set_feature_size, 50),
            nn.ELU(inplace=True),
            nn.Linear(50, set_latent_size),
            nn.ELU(inplace=True),
            nn.Linear(set_latent_size, set_latent_size)
        )

        self.set_regressor = nn.Sequential(
            nn.Linear(set_latent_size, set_latent_size),
            nn.ELU(inplace=True),
            nn.Linear(set_latent_size, set_latent_size),
            nn.ELU(inplace=True),
            nn.Linear(set_latent_size, img_size * img_size),
            nn.ELU(inplace=True)
        )

        
        padding = int((kernel_size-1)/2)
        alpha = 0.2
        init_cnn_layers = []
        init_cnn_layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        init_cnn_layers.append(nn.LeakyReLU(negative_slope=alpha,inplace=True))
        for _ in range(num_init_CNN_layers-1):
            init_cnn_layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            init_cnn_layers.append(nn.ReLU(inplace=True))

        self.init_cnn = nn.Sequential(*init_cnn_layers)


        post_cnn_layers = []
        post_cnn_layers.append(nn.Conv2d(in_channels=features+1, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        post_cnn_layers.append(nn.ReLU(inplace=True))
        for _ in range(num_post_CNN_layers-2):
            post_cnn_layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            post_cnn_layers.append(nn.ReLU(inplace=True))

        post_cnn_layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.post_cnn = nn.Sequential(*post_cnn_layers)


        

    def forward(self, img, feats):
        img = self.init_cnn(img)
        feats = self.set_feature_extractor(feats)
        feats = feats.sum(dim=1)
        feats = self.set_regressor(feats)
        feats = feats.view(-1, 1, self.img_size, self.img_size)
        img = torch.cat((feats, img), dim = 1)
        out = self.post_cnn(img)

        return out


class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=9, kernel_size=3, features=100):
        super(DnCNN, self).__init__()
        padding = int((kernel_size-1)/2)
        alpha = 0.2
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.LeakyReLU(negative_slope=alpha,inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

class PatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size=50):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            max_patch_loss = 0
            # calculate loss for each patch of the image
            for j in range(list(output_patches.size())[0]):
                for k in range(list(output_patches.size())[1]):
                    max_patch_loss = max(max_patch_loss, f.l1_loss(output_patches[j][k], target_patches[j][k]))
            avg_loss+=max_patch_loss
        avg_loss/=len(output)
        return avg_loss;

class WeightedPatchLoss(nn.Module):
    def __initII(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PatchLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, output, target, patch_size):
        avg_loss = 0
        for i in range(len(output)):
            # split output and target images into patches
            output_patches = output[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            target_patches = target[i].unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            weighted_loss = 0
            # calculate loss for each patch of the image
            for j in range(list(output_patches.size())[0]):
                for k in range(list(output_patches.size())[1]):
                    weighted_loss += f.l1_loss(output_patches[j][k],target_patches[j][k]) * torch.mean(target_patches[j][k])
            avg_loss+=weighted_loss/torch.mean(target[i])
        avg_loss/=len(output)
        return avg_loss;


if __name__=="__main__":
    criterion = PatchLoss()
    dtype = torch.FloatTensor

    x = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    y = Variable(torch.randn(100, 100).type(dtype), requires_grad=False)
    loss = criterion(x, y, 10)
    net = DnCNN()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
