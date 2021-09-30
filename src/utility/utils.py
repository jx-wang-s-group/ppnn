import torch


def build_FD_Central_filter_2D(c:list):
    length = c.__len__()
    c = torch.tensor(c)
    x = torch.zeros([length,length])
    x[:,length//2] = c
    x[length//2,:] += c
    return x