import os
import logging
import numpy as np
try:
    import mxnet as mx
except ImportError:
    mx = None
try:
    from PIL import Image
except ImportError:
    Image = None

from .utils import makenp, save_image


def _make_tsv(metadata, save_path):
    metadata = [str(x) for x in metadata]
    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata:
            f.write(x + '\n')


def _make_sprite(img_labels, save_path):
    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    img_labels_shape = img_labels.shape
    nrow = int(np.ceil(np.sqrt(img_labels_shape[0])))

    img_labels = makenp(img_labels)
    # augment images so that #images equals nrow*nrow
    img_labels = np.concatenate((img_labels,
                                 np.random.normal(loc=0, scale=1,
                                                  size=((nrow*nrow-img_labels_shape[0],)+img_labels_shape[1:])) * 255),
                                axis=0)
    save_image(img_labels, os.path.join(save_path, 'sprite.png'), nrow=nrow, padding=0)


def _append_pbtxt(metadata, label_img, save_path, global_step, tag):
    with open(os.path.join(save_path, 'projector_config.pbtxt'), 'a') as f:
        # step = os.path.split(save_path)[-1]
        s = 'embeddings {\n'
        s += 'tensor_name: "{}:{}"\n'.format(tag, global_step)
        s += 'tensor_path: "{}"\n'.format(os.path.join(global_step, 'tensors.tsv'))
        if metadata is not None:
            s += 'metadata_path: "{}"\n'.format(os.path.join(global_step, 'metadata.tsv'))
        if label_img is not None:
            if label_img.ndim != 4:
                logging.warn('expected 4D sprite image in the format NCHW, while received image ndim=%d,'
                             ' skipping saving sprite image info'
                             % label_img.ndim)
            else:
                s += 'sprite {\n'
                s += 'image_path: "{}"\n'.format(os.path.join(global_step, 'sprite.png'))
                s += 'single_image_dim: {}\n'.format(label_img.shape[3])
                s += 'single_image_dim: {}\n'.format(label_img.shape[2])
                s += '}\n'
        s += '}\n'
        f.write(s)


def _save_ndarray_to_file(data, file_path):
    if isinstance(data, np.ndarray):
        data_list = data.tolist()
    elif mx is not None and isinstance(data, mx.nd.NDArray):
        data_list = data.asnumpy().tolist()
    else:
        raise ValueError('only supports saving numpy.ndarray and MXNet NDArray to file if MXNet is installed,'
                         ' while received type=%s' % str(type(data)))
    with open(os.path.join(file_path, 'tensors.tsv'), 'w') as f:
        for x in data_list:
            x = [str(i) for i in x]
            f.write('\t'.join(x) + '\n')