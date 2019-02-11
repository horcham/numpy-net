import numpy as np

def im2col(X, filter, stride=1, padH=0, padW=0):
    N, C, H, W = X.shape
    filter_h, filter_w, filter_c, filter_c2 = filter.shape
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
            cols[:, :, h, :] = crop
    cols = cols.reshape([cols.shape[0], cols.shape[1] * cols.shape[2], -1])
    cols = cols.transpose([0, 2, 1])
    return cols

