from __future__ import print_function

import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from mxnet import gluon
from tensorboardX import SummaryWriter

# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')

opt = parser.parse_args()


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32)/255
    return data, label


train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST('./data', train=True, transform=transformer),
    batch_size=opt.batch_size, shuffle=True, last_batch='discard')

initialized = False
embedding = None
labels = None
images = None

for i, (data, label) in enumerate(train_data):
    if i >= 20:
        break
    if initialized:
        embedding = np.concatenate((embedding, data.flatten().asnumpy()), axis=0)
        labels = np.concatenate((labels, label.asnumpy().flatten()), axis=0)
        images = np.concatenate((images, data.asnumpy().reshape(opt.batch_size, 1, 28, 28)), axis=0)
    else:
        embedding = data.flatten().asnumpy()
        labels = label.asnumpy().flatten()
        images = data.asnumpy().reshape(opt.batch_size, 1, 28, 28)
        initialized = True

logdir = "./logs"
summary_writer = SummaryWriter(logdir=logdir)
summary_writer.add_embedding(embedding, labels=labels, images=images, tag='mnist')