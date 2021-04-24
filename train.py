# from __future__ import division
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tqdm import tqdm

from nets.frcnn import get_model
from nets.data_gen import Generator, get_img_output_length
from nets.frcnn_loss import class_loss_cls, class_loss_regr, cls_loss, smooth_l1
from faster_utils.anchors import get_anchors
from faster_utils.config import Config
from faster_utils.roi_helpers import calc_iou
from faster_utils.utils_new import BBoxUtility

# 禁用gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def write_log(callback, names, logs, batch_no):
    with callback.as_default():
        for name, value in zip(names, logs):
            tf.summary.scalar(name, value, step=batch_no)
            callback.flush()


def model_train(model_rpn, inputs_img, true_boxes, rpn_true_boxes):
    '''
    模型训练过程,分两步进行:
    1. 图片输入rpn预测网络获得每个先验框应有的变化量,和每个先验框是否包含物体置信度
    2. rpn预测结果与先验框组合,获得建议框和每个建议框包含物体的置信度(就是上面的置信度,只是提取了最好的一些,之前置信度对应
        先验框变化量, 现在对应着处理后的建议框)
    3. 建议框与真实框进行iou计算,根据阈值确定正负样本,并根据iou值确定哪个建议框负责预测哪个真实框. 并算出该建议框要想获得真实
        框坐标和宽高应该有的偏移量和宽高比
    4. 将过滤后的建议框坐标输入分类网络进行最后训练,获得最终预测类别和建议框应有的变化量
    Parameters
    ----------
    model_rpn: rpn网络模型
    inputs_img:图片
    true_boxes:真实框坐标
    rpn_true_boxes

    Returns
    -------
    '''

    rpn_predict = model_rpn.predict_on_batch(inputs_img)

    # (batch_size,600,600,3)
    height, width, _ = np.shape(inputs_img[0])
    feature_w, feature_h = get_img_output_length(width, height)
    anchors = get_anchors([feature_w, feature_h], width, height)
    # rpn预测结果与先验框，共同确定建议框，并提取前300个
    results = bbox_util.detection_out_rpn(rpn_predict, anchors)

    roi_inputs, out_classes, out_regrs = list(), list(), list()

    # 遍历一个batch_size中的每张图片
    for j in range(len(inputs_img)):
        # rpn模型预测的调整后的建议框坐标
        # 取出每张图片所有建议框的坐标
        proposal_boxes = results[j][:, 1:]
        '''
        pos_neg_loc:调整正负样本数量后的建议框坐标：x,y,w,h
        labels:建议框应该预测的类别值（已one_hot编码）
        true_boxes_offset：建议框若要调整到真实框位置，应该有的偏移量和宽高比，x,y,w,h
        '''
        pos_neg_loc, labels, true_boxes_offset = calc_iou(proposal_boxes,
                                                          config,
                                                          true_boxes[j],
                                                          NUM_CLASSES)
        roi_inputs.append(pos_neg_loc)
        out_classes.append(labels)
        out_regrs.append(true_boxes_offset)

    loss_class = model_all.train_on_batch([inputs_img, np.array(roi_inputs)],
                                          [rpn_true_boxes[0], rpn_true_boxes[1],
                                           np.array(out_classes), np.array(out_regrs)])

    return loss_class


def fit_one_epoch(model_rpn, model_all, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, callback):
    '''
    使用train_on_batch进行更精细的训练,因为rpn层每次训练后的输出是分类网络的输入
    '''
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0

    val_toal_loss = 0
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for i, batch in enumerate(gen):
            if i >= epoch_size:
                break

            inputs_img, rpn_true_boxes, true_boxes = batch[0], batch[1], batch[2]

            loss_class = model_train(model_rpn, inputs_img, true_boxes, rpn_true_boxes)

            write_log(callback,
                      ['total_loss', 'rpn_cls_loss', 'rpn_reg_loss', 'detection_cls_loss', 'detection_reg_loss'],
                      loss_class, i)

            rpn_cls_loss += loss_class[1]
            rpn_loc_loss += loss_class[2]
            roi_cls_loss += loss_class[3]
            roi_loc_loss += loss_class[4]
            total_loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss

            pbar.set_postfix(**{'total': total_loss / (i + 1),
                                'rpn_cls': rpn_cls_loss / (i + 1),
                                'rpn_loc': rpn_loc_loss / (i + 1),
                                'roi_cls': roi_cls_loss / (i + 1),
                                'roi_loc': roi_loc_loss / (i + 1)})
            pbar.update(1)

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for i, batch in enumerate(genval):
            if i >= epoch_size_val:
                break
            inputs_img, rpn_true_boxes, true_boxes = batch[0], batch[1], batch[2]

            loss_class = model_train(model_rpn, inputs_img, true_boxes, rpn_true_boxes)

            val_toal_loss += loss_class[0]
            pbar.set_postfix(**{'total': val_toal_loss / (i + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print(f'Total Loss: {total_loss / (epoch_size + 1)} || Val Loss: {val_toal_loss / (epoch_size_val + 1)} ')
    print('Saving state, iter:', str(epoch + 1))

    model_all.save_weights(f'logs/Epoch{(epoch + 1)}-Total_Loss{total_loss / (epoch_size + 1)}-'
                           f'Val_Loss{val_toal_loss / (epoch_size_val + 1)}.h5')

    return


if __name__ == "__main__":
    config = Config()
    #   修改成所需要区分的类的个数+1。
    NUM_CLASSES = 21
    # 有600,800两种shape可选,多次训练测试后发现800,800,3更优
    input_shape = [800, 800, 3]
    Batch_size = 2

    annotation_path = 'settings/2007_train.txt'

    # 正常情况下val_split=0.1, 验证集和训练集的比例为1:9
    val_split = 0.5
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    #  训练参数的设置
    callback = tf.summary.create_file_writer("logs")

    bbox_util = BBoxUtility(iou_threshold=config.rpn_max_iou,
                            ignore_threshold=config.rpn_min_iou,
                            top_k=config.num_RPN_train_pre)

    # 创建训练和测试用的生成器对象
    gen = Generator(bbox_util, lines[:num_train],
                    NUM_CLASSES, Batch_size, input_shape=input_shape[:2]).generate()
    gen_val = Generator(bbox_util, lines[num_train:],
                        NUM_CLASSES, Batch_size, input_shape=input_shape[:2]).generate()

    # 划分训练和测试步数
    epoch_size = num_train // Batch_size
    epoch_size_val = num_val // Batch_size

    # 创建模型并加载预训练权重
    model_rpn, model_all = get_model(config, NUM_CLASSES)
    base_net_weights = "model_data/voc_weights.h5"
    model_all.load_weights(base_net_weights, by_name=True)

    # 根据自身需要,选择冻结的层数. 之后的精细训练再放开,可保持预训练权重不能破坏
    freeze_layers = int(0.618 * len(model_all.layers))
    print('冻结层数量:', freeze_layers)
    for i in range(freeze_layers):
        model_all.layers[i].trainable = False

    # Init_Epoch为起始世代
    # Interval_Epoch为中间训练的世代
    # Epoch总训练世代
    # 提示OOM或者显存不足请调小Batch_size
    if True:
        lr = 1e-4
        Init_Epoch = 0
        Interval_Epoch = 1

        model_all.compile(optimizer=keras.optimizers.Adam(lr=lr),
                          loss={
                              'classification': cls_loss(),
                              'regression': smooth_l1(),
                              f'dense_class_{NUM_CLASSES}': class_loss_cls,
                              f'dense_regress_{NUM_CLASSES}': class_loss_regr(NUM_CLASSES - 1)
                          })

        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model_rpn, model_all, epoch, epoch_size,
                          epoch_size_val, gen, gen_val, Interval_Epoch, callback)

            lr = lr * 0.92
            # 每个epoch,学习率都要变小
            K.set_value(model_all.optimizer.lr, lr)

    if True:
        lr = 1e-5
        Interval_Epoch = 1
        Epoch = 2

        # 解冻训练
        for i in range(freeze_layers):
            model_all.layers[i].trainable = True

        # 学习率改变,模型要重新装配
        model_all.compile(optimizer=keras.optimizers.Adam(lr=lr),
                          loss={
                              'classification': cls_loss(),
                              'regression': smooth_l1(),
                              f'dense_class_{NUM_CLASSES}': class_loss_cls,
                              f'dense_regress_{NUM_CLASSES}': class_loss_regr(NUM_CLASSES - 1)
                          })

        # 接着前面训练的epoch数量继续.
        for epoch in range(Interval_Epoch, Epoch):
            fit_one_epoch(model_rpn, model_all, epoch, epoch_size,
                          epoch_size_val, gen, gen_val, Epoch, callback)

            lr = lr * 0.92
            K.set_value(model_all.optimizer.lr, lr)
