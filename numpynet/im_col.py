import numpy as np
# import minpy.numpy as np
import sys
from numpynet.variable import *

def im2col(X, filter_h, filter_w, stride=1, padH=0, padW=0):
    N, C, H, W = X.shape
    out_h = (H + 2 * padH - filter_h) // stride + 1
    out_w = (W + 2 * padW - filter_w) // stride + 1
    img = np.pad(X, [(0, 0), (0, 0), (padH, padH), (padW, padW)], 'constant')
    cols = np.zeros([N, C, filter_h * filter_w, out_h * out_w])
    for h in range(filter_h):
        h_max = h + stride * out_h
        for w in range(filter_w):
            w_max = w + stride * out_w
            crop = img[:, :, h:h_max:stride, w:w_max:stride]
            crop = crop.reshape([img.shape[0], img.shape[1], -1])
            cols[:, :, h*filter_w+w, :] = crop
    cols = cols.reshape([cols.shape[0], cols.shape[1] * cols.shape[2], -1])
    cols = cols.transpose([0, 2, 1])
    return cols


def col2im(X, filter_h, filter_w, image_size, stride=1):
    out_N, out_C, out_H, out_W = image_size
    img = np.zeros(image_size)
    weight = np.zeros(image_size)

    X = X.transpose([0, 2, 1])
    X = X.reshape([X.shape[0], out_C, X.shape[1] / out_C, X.shape[2]])
    k = 0
    for h in range(0, img.shape[2] - filter_h + 1, stride):
        for w in range(0, img.shape[3] - filter_w + 1, stride):
            # print(img[:, :, h:h + filter_h, w:w + filter_w].shape)
            # print(X[:, :, :, k].shape)
            img[:, :, h:h + filter_h, w:w + filter_w] += X[:, :, :, k].reshape([X.shape[0], X.shape[1], filter_h, filter_w])
            weight[:, :, h:h + filter_h, w:w + filter_w] += np.ones(([X.shape[0], X.shape[1], filter_h, filter_w]))
            k += 1
    return img / (weight + 1e-6)

