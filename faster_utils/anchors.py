import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from faster_utils.config import Config

config = Config()


def generate_anchors(sizes=None, ratios=None):
    '''
    生成不同大小的先验框,生成9个不同尺度的框,对应共享特征层
    上每个网格点,前三个是正方形. 后六个是长方形
    '''
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # 如[-64,-64,64,64]  相对于原点(0,0)
    return anchors


def shift(shape, anchors, stride=config.rpn_stride):
    # [0,1,2,...,37],其中每个值加上0.5
    # rpn_stride=16
    # 将图片划分成38x3网格，获得划分后的网格中心坐标，如shift_x=[8,24,....]
    shift_x = (np.arange(0, shape[0], dtype=K.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=K.floatx()) + 0.5) * stride

    # 获得网格中心点
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)

    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    # 网格中心点坐标,+-先验框宽/高的0.5倍, 获得先验框坐标.
    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + \
                      np.array(np.reshape(shifts, [k, 1, 4]), K.floatx())
    # 二维数组，获得所有的先验框
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


def get_anchors(shape, width, height):
    # 生成的框是针对600x600图片的
    anchors = generate_anchors()
    # 将600x600图片划分成38x38网格,并未每个网格中心分配9个先验框,并取得这些先验框坐标
    network_anchors = shift(shape, anchors)

    # 将先验框坐标归一化
    network_anchors[:, [0, 2]] = network_anchors[:, [0, 2]] / width
    network_anchors[:, [1, 3]] = network_anchors[:, [1, 3]] / height
    # 固定在(0,1),不超出图片范围
    network_anchors = np.clip(network_anchors, 0, 600)

    return network_anchors


if __name__ == '__main__':
    res = get_anchors([38, 38], 600, 600)
    print(res)
