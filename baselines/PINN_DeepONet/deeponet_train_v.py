import numpy as np
import deepxde as dde
import torch
import torch.nn as nn
from math import ceil
import torch.nn.functional as F
import os


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


path = './model'
mkdir(path)

NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}


class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class Encoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(Encoder, self).__init__()
        self.reshape1 = Reshape(3, 256, 256)
        self.convblock1_1 = ConvBlock(3, dim1, kernel_size=3, stride=2, padding=1)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=2, padding=1)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5_1 = ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1)
        self.convblock5_2 = ConvBlock(dim5, dim5)
        self.convblock6 = ConvBlock(dim5, 2*dim5, kernel_size=8, padding=0)
        self.reshape2 = Reshape(2*dim5)
        self.convblock7 = nn.Linear(2*dim5, dim5)


    def forward(self, x):
        # Encoder Part
        x = self.reshape1(x)
        x = self.convblock1_1(x)  # (None, 32, 128, 128)
        x = self.convblock1_2(x)  # (None, 32, 128, 128)
        x = self.convblock2_1(x)  # (None, 64, 64, 64)
        x = self.convblock2_2(x)  # (None, 64, 64, 64)
        x = self.convblock3_1(x)  # (None, 128, 32, 32)
        x = self.convblock3_2(x)  # (None, 128, 32, 32)
        x = self.convblock4_1(x)  # (None, 256, 16, 16)
        x = self.convblock4_2(x)  # (None, 256, 16, 16)
        x = self.convblock5_1(x)  # (None, 256, 8, 8)
        x = self.convblock5_2(x)  # (None, 256, 8, 8)
        x = self.convblock6(x)  # (None, 512, 1, 1)
        x = self.reshape2(x)
        x = self.convblock7(x)

        return x


class TripleCartesianProd(dde.data.Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`). The mini-batch
            is only applied to `N1`.
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if len(X_train[0]) * len(X_train[1]) != y_train.size:
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if len(X_test[0]) * len(X_test[1]) != y_test.size:
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = dde.data.BatchSampler(len(X_train[0]), shuffle=True)
        self.train_timestep_sampler = dde.data.BatchSampler(100,shuffle=True)
        self.test_sampler = dde.data.BatchSampler(len(X_test[0]), shuffle=True)
        self.timestep_batch_size = 20

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None, ):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        indices_timestep = self.train_timestep_sampler.get_next(self.timestep_batch_size)
        size = self.train_y.shape[0]
        return (
            self.train_x[0][indices],
            self.train_x[1].reshape(100,256,256,3)[indices_timestep,:].reshape(self.timestep_batch_size*256*256, 3),
        ), self.train_y.reshape(size,100,256,256)[indices,:][:,indices_timestep,:].reshape(batch_size,-1)

    def test(self):
        batch_size_test = 16
        indices_test = self.test_sampler.get_next(batch_size_test)
        indices_timestep_test = self.train_timestep_sampler.get_next(self.timestep_batch_size)
        size_test = self.test_y.shape[0]
        return (
            self.test_x[0][indices_test],
            self.test_x[1].reshape(100,256,256,3)[indices_timestep_test,:].reshape(self.timestep_batch_size*256*256, 3),
        ), self.test_y.reshape(size_test,100,256,256)[indices_test, :][:,indices_timestep_test,:].reshape(batch_size_test ,self.timestep_batch_size*256*256)


def main():
    X_train_branch = np.load('X_train_branch.npy')[:, :, :-1, :-1].reshape(216, 3 * 256 * 256).astype(np.float32)
    X_test_branch = np.load('X_test_branch.npy')[:, :, :-1, :-1].reshape(100, 3 * 256 * 256).astype(np.float32)
    y_train = torch.load("training_data.pth", map_location=torch.device('cpu'))['u'].numpy()[:, 1:, 1, :-1, :-1].reshape(216, 100 * 1 * 256 * 256).astype(np.float32)
    y_test = torch.load('GroundTruth_test.pth', map_location=torch.device('cpu'))['u'].numpy()[:, 1:101, 1, :-1, :-1].reshape(100, 100 * 1 * 256 * 256).astype(np.float32)
    txy = np.array([[t, x, y] for t in np.linspace(0, 1, 101)[1:] for x in np.linspace(0, 1, 257)[:-1] for y in
                    np.linspace(0, 1, 257)[:-1]]).astype(np.float32)
    X_train = (X_train_branch, txy)
    X_test = (X_test_branch, txy)
    data = TripleCartesianProd(X_train, y_train, X_test, y_test)

    m = 3 * 256 ** 2
    activation = "relu"
    branch = Encoder()

    net = dde.nn.pytorch.DeepONetCartesianProd(
        [m, branch], [17, 512, 512, 512, 512], activation, "Glorot normal", regularization=['l2', 1e-5]
    )

    def periodic(x):
        return torch.cat([x[:, 0:1],
                          torch.cos(1 * x[:, 1:2] * 2 * np.pi), torch.sin(1 * x[:, 1:2] * 2 * np.pi),
                          torch.cos(2 * x[:, 1:2] * 2 * np.pi), torch.sin(2 * x[:, 1:2] * 2 * np.pi),
                          torch.cos(3 * x[:, 1:2] * 2 * np.pi), torch.sin(3 * x[:, 1:2] * 2 * np.pi),
                          torch.cos(4 * x[:, 1:2] * 2 * np.pi), torch.sin(4 * x[:, 1:2] * 2 * np.pi),
                          torch.cos(1 * x[:, 2:3] * 2 * np.pi), torch.sin(1 * x[:, 2:3] * 2 * np.pi),
                          torch.cos(2 * x[:, 2:3] * 2 * np.pi), torch.sin(2 * x[:, 2:3] * 2 * np.pi),
                          torch.cos(3 * x[:, 2:3] * 2 * np.pi), torch.sin(3 * x[:, 2:3] * 2 * np.pi),
                          torch.cos(4 * x[:, 2:3] * 2 * np.pi), torch.sin(4 * x[:, 2:3] * 2 * np.pi)], 1)
    net.apply_feature_transform(periodic)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=5e-4,
        loss=dde.losses.mean_l2_relative_error,
        decay=("step", 5000, 0.9),
        metrics=["mean l2 relative error"]
    )
    checker = dde.callbacks.ModelCheckpoint("model/model_v", save_better_only=False, period=100)
    losshistory, train_state = model.train(epochs=30000, batch_size=16, display_every=100, callbacks=[checker])
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)


if __name__ == "__main__":
    main()

