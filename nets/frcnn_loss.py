import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# 使用闭包的形式返回损失函数
def cls_loss():
    def _cls_loss(y_true, y_pred):
        #   y_true,y_pred shape=[batch_size, num_anchor, 1]
        #   获得无需忽略的所有样本,-1 是需要忽略的, 0 是背景, 1 是存在目标
        no_ignore_mask = tf.where(K.not_equal(y_true, -1))
        true_label = tf.gather_nd(y_true, no_ignore_mask)
        predict_label = tf.gather_nd(y_pred, no_ignore_mask)

        cls_loss = K.binary_crossentropy(true_label, predict_label)
        cls_loss = K.sum(cls_loss)

        #   进行标准化
        normalizer_no_ignore = K.cast(K.shape(no_ignore_mask)[0], K.floatx())
        normalizer_no_ignore = K.maximum(K.cast_to_floatx(1.0), normalizer_no_ignore)

        # 总的loss
        loss = cls_loss / normalizer_no_ignore
        return loss

    return _cls_loss


def smooth_l1(sigma=1.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        #   y_true [batch_size, num_anchor, 4+1]
        #   y_pred [batch_size, num_anchor, 4]
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        indices = tf.where(K.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算smooth L1损失
        regression_diff = regression - regression_target
        regression_diff = K.abs(regression_diff)
        regression_loss = tf.where(K.less(regression_diff, 1.0 / sigma_squared),
                                   0.5 * sigma_squared * K.pow(regression_diff, 2),
                                   regression_diff - 0.5 / sigma_squared
                                   )

        # 将所获得的loss除上正样本的数量
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())
        regression_loss = K.sum(regression_loss) / normalizer
        return regression_loss

    return _smooth_l1


def class_loss_regr(num_classes):
    '''
    计算类别对应的框的损失
    Parameters
    ----------
    num_classes

    Returns
    -------

    '''
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * K.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(
            epsilon + y_true[:, :, :4 * num_classes])
        return loss

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
    return loss
