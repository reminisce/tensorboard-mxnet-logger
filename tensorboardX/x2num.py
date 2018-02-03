import numpy as np
try:
    import mxnet as mx
except ImportError:
    mx = None


def _makenp(x, modality=None):
    # if already numpy, return
    if isinstance(x, np.ndarray):
        if modality == 'IMG' and x.dtype == np.uint8:
            return x.astype(np.float32) / 255.0
        return x
    elif np.isscalar(x):
        return np.array([x])
    elif mx is not None and isinstance(x, mx.nd.NDArray):
        return _mxnet_np(x, modality)
    else:
        raise TypeError('_makenp only accepts input types of numpy.ndarray, scalar,'
                        ' and MXNet NDArray if MXNet has been installed,'
                        ' while received type=%s' % str(type(x)))


def _mxnet_np(x, modality):
    assert mx is not None
    assert isinstance(x, mx.nd.NDArray)
    x = x.asnumpy()
    if modality == 'IMG':
        x = _prepare_image(x)
    return x


def _make_grid(img, ncols=8):
    assert isinstance(img, np.ndarray), 'plugin error, should pass numpy array here'
    assert img.ndim == 4 and img.shape[1] == 3
    nimg = img.shape[0]
    h = img.shape[2]
    w = img.shape[3]
    ncols = min(nimg, ncols)
    nrows = int(np.ceil(float(nimg) / ncols))
    canvas = np.zeros((3, h * nrows, w * ncols))
    i = 0
    for y in range(nrows):
        for x in range(ncols):
            if i >= nimg:
                break
            canvas[:, y * h:(y + 1) * h, x * w:(x + 1) * w] = img[i]
            i = i + 1
    return canvas


def _prepare_image(img):
    assert isinstance(img, np.ndarray), 'plugin error, should pass numpy array here'
    assert img.ndim == 2 or img.ndim == 3 or img.ndim == 4
    if img.ndim == 4:  # NCHW
        if img.shape[1] == 1:  # N1HW
            img = np.concatenate((img, img, img), 1)  # N3HW
        assert img.shape[1] == 3
        img = _make_grid(img)  # 3xHxW
    if img.ndim == 3 and img.shape[0] == 1:  # 1xHxW
        img = np.concatenate((img, img, img), 0)  # 3xHxW
    if img.ndim == 2:  # HxW
        img = np.expand_dims(img, 0)  # 1xHxW
        img = np.concatenate((img, img, img), 0)  # 3xHxW
    img = img.transpose(1, 2, 0)

    return img
