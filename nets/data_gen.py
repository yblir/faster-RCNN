import random
from random import shuffle
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from faster_utils.anchors import get_anchors


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_new_img_size(width, height, img_min_side=600):
    '''
    将最短边设为600,等比例缩放
    Parameters
    ----------
    width
    height
    img_min_side

    Returns
    -------

    '''
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3, 1, 0, 0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width), get_output_length(height)


class Generator(object):
    def __init__(self, bbox_util, train_lines, num_classes, Batch_size, input_shape=[600, 600], num_regions=256):
        self.bbox_util = bbox_util
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.num_classes = num_classes
        self.Batch_size = Batch_size
        self.input_shape = input_shape
        self.num_regions = num_regions

    def get_random_data(self, annotation_line, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        w, h = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]

            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            # 使用len(box),很谨慎,防止赋值时出现数量不对应的情况 box=[x_min,y_min,x_max,y_max,c],未归一化
            box_data[:len(box)] = box

        return image_data, box_data

    def generate(self):
        while True:
            shuffle(self.train_lines)

            inputs = list()
            contain_object = list()
            assignment_info = list()
            true_boxes = list()

            for line in self.train_lines:
                # img:增强后的图片, y: 真实框位置+类别
                img, true_box = self.get_random_data(line)
                height, width, _ = np.shape(img)

                if len(true_box) > 0:
                    # 真实框位置归一化
                    true_box[:, [0, 2]] = true_box[:, [0, 2]] / width
                    true_box[:, [1, 3]] = true_box[:, [1, 3]] / height

                # 取得所有先验框,已归一化
                anchors = get_anchors(get_img_output_length(width, height), width, height)

                #   :, :4 的内容为rpn网络应该有的预测结果, 已完成归一化处理
                #   :,  4 的内容为先验框是否包含物体，默认为背景,0:背景,-1:负样本,1:正样本
                assignment = self.bbox_util.assign_boxes(true_box, anchors)

                # 取出当前先验框是否包含物体
                contain_obj = assignment[:, 4]
                regression = assignment[:, :]

                #   对正样本与负样本进行筛选，训练样本总和为256
                mask_pos = contain_obj[:] > 0
                num_pos = len(contain_obj[mask_pos])

                # 若正样本大于样本总数的一半,将多于一半的正样本忽略
                if num_pos > self.num_regions / 2:
                    val_locs = random.sample(range(num_pos), int(num_pos - self.num_regions / 2))
                    temp_contain_obj = contain_obj[mask_pos]
                    temp_regression = regression[mask_pos]
                    temp_contain_obj[val_locs] = -1
                    temp_regression[val_locs, -1] = -1
                    contain_obj[mask_pos] = temp_contain_obj
                    regression[mask_pos] = temp_regression

                # 重新计算正负样本数量
                mask_neg = contain_obj[:] == 0
                num_neg = len(contain_obj[mask_neg])
                mask_pos = contain_obj[:] > 0
                num_pos = len(contain_obj[mask_pos])

                # 若正负样本总数大于256,就将多出的负样本忽略
                if num_neg + num_pos > self.num_regions:
                    val_locs = random.sample(range(num_neg), int(num_neg + num_pos - self.num_regions))
                    temp_contain_obj = contain_obj[mask_neg]
                    temp_contain_obj[val_locs] = -1
                    # 将多出来的负样本,设为不包含物体,忽略,平衡正负样本数量
                    contain_obj[mask_neg] = temp_contain_obj

                inputs.append(img)
                # 是否包含物体
                contain_object.append(np.reshape(contain_obj, [-1, 1]))
                # 应该有的预测框坐标+是否包含物体
                assignment_info.append(np.reshape(regression, [-1, 5]))
                # 真实框的坐标: 真实框位置+类别
                true_boxes.append(true_box)

                if len(inputs) == self.Batch_size:
                    inputs = np.array(inputs)
                    rpn_true_boxes = [np.array(contain_object, np.float32), np.array(assignment_info, np.float32)]
                    # 图片,建议框网络应该有的预测结果,真实框
                    yield preprocess_input(inputs), rpn_true_boxes, true_boxes
                    # 一个batch_size清空结果,重新来过
                    inputs = list()
                    contain_object = list()
                    assignment_info = list()
                    true_boxes = list()
