import os
os.environ["DDEBACKEND"] = "pytorch"
import numpy as np
import deepxde as dde
import torch
import torch.nn as nn
import time
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


class Encoder(torch.nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.reshape1 = Reshape(3, 256, 256)
        self.convblock1_1 = ConvBlock(3, 16, kernel_size=3, stride=2, padding=1)
        self.convblock1_2 = ConvBlock(16, 16)
        self.convblock2_1 = ConvBlock(16, 32, kernel_size=3, stride=2, padding=1)
        self.convblock2_2 = ConvBlock(32, 32)
        self.convblock3_1 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.convblock3_2 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)
        self.convblock4_1 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.convblock5 = ConvBlock(128, 128, kernel_size=5, stride=3, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, 256)

    def forward(self, x):
        # Encoder Part
        x = self.reshape1(x)
        x = self.convblock1_1(x)  # (None, 16, 128, 128)
        x = self.convblock1_2(x)  # (None, 16, 128, 128)
        x = self.convblock2_1(x)  # (None, 32, 64, 64)
        x = self.convblock2_2(x)  # (None, 32, 64, 64)
        x = self.convblock3_1(x)  # (None, 64, 32, 32)
        x = self.convblock3_2(x)  # (None, 64, 16, 16)
        x = self.convblock4_1(x)  # (None, 128, 8, 8)
        x = self.convblock5(x)  # (None, 128, 2, 2)
        x = self.flatten(x)  # (None, 512)
        x = self.linear1(x)  # (None, 256)
        return x


batch_time = 200
traj = 100
y_true = torch.load('/home/xinyang/GroundTruth_test.pth', map_location=torch.device('cpu'))['u'].numpy()[:traj]
X_test_branch = np.load('/home/xinyang/X_test_branch.npy')[:traj, :, :-1, :-1].reshape(traj, 3 * 256 * 256).astype(np.float32)
m = 3 * 256 ** 2
activation = "relu"
branch = Encoder()

net = dde.nn.pytorch.DeepONetCartesianProd(
    [m, branch], [17, 128, 128, 128, 256], activation, "Glorot normal", regularization=['l2', 1e-6]
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

y_test = y_true[0:1, 1:1 + batch_time, 0, :-1, :-1].reshape(1, batch_time * 256 * 256)
txy = np.array([[t, x, y] for t in np.linspace(0, 2, 201)[1:1 + batch_time]
                for x in np.linspace(0, 1, 257)[:-1] for y in np.linspace(0, 1, 257)[:-1]]).astype(np.float32)
X_test = (X_test_branch[0:1], txy)
data = dde.data.TripleCartesianProd(X_test, y_test, X_test, y_test)
Model = dde.Model(data, net)
Model.compile("adam", lr=5e-4, loss=dde.losses.mean_l2_relative_error, decay=("step", 5000, 0.9))


def u_pred(model, trunk):
    y_pred = []
    times = 0
    for i in range(200 // batch_time):
        x_test = (X_test_branch, trunk)
        model.restore(f"model_u.pt", verbose=1)
        time_start = time.time()
        y_pred_i = model.predict(x_test)
        times += time.time() - time_start
        y_pred.append(y_pred_i)
        torch.cuda.empty_cache()
    print(f"Inference time for u of {traj} trajectories: {times}s")
    y_pred = np.concatenate(y_pred, axis=1)
    return y_pred, times


def v_pred(model, trunk):
    y_pred = []
    times = 0
    for i in range(200 // batch_time):
        x_test = (X_test_branch, trunk)
        model.restore(f"model_v.pt", verbose=1)
        time_start = time.time()
        y_pred_i = model.predict(x_test)
        times += time.time() - time_start
        y_pred.append(y_pred_i)
        torch.cuda.empty_cache()
    print(f"Inference time for v of {traj} trajectories: {times}s")
    y_pred = np.concatenate(y_pred, axis=1)
    return y_pred, times


def error(x, gt):
    return np.sqrt(((x - gt) ** 2).mean(axis=(0, 2, 3, 4)) / (gt ** 2).mean(axis=(0, 2, 3, 4)))


def main():
    u_prediction, time_u = u_pred(Model, txy)
    u_prediction = u_prediction.reshape([traj, 200, 1, 256, 256])
    v_prediction, time_v = v_pred(Model, txy)
    v_prediction = v_prediction.reshape([traj, 200, 1, 256, 256])
    print(f"Total inference time of {traj} trajectories: {time_u + time_v}s")
    y_prediction = np.concatenate([u_prediction, v_prediction], axis=2)
    y_prediction = np.concatenate([y_prediction, y_prediction[:, :, :, 0:1, :]], axis=3)
    y_prediction = np.concatenate([y_prediction, y_prediction[:, :, :, :, 0:1]], axis=4)
    errors = error(y_prediction[:, :, :, :, :], y_true[:, 1:, :, :, :])
    print(f"average_error: {errors.mean()}")
    np.savetxt("errors_test.txt", errors)
    # np.save("prediction.npy", y_prediction)


if __name__ == "__main__":
    main()