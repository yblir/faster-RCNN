import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class RoiPoolingConv(Layer):
    def __init__(self, **kwargs):
        super(RoiPoolingConv, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        # base_layers:共享特征层, shape=[None,38,38,1024],None表示batch_size
        img, rois, pool_size = inputs  # 共享特征层,建议框
        nb_channels = K.int_shape(img)[3]

        # roi_input:[None,None,4],第一个None表示batch_size, 第二个None是shape中的,表示
        # 一张图片有多少个建议框,4表示这些建议框的的坐标
        batch_size = tf.shape(rois)[0]
        num_rois = tf.shape(rois)[1]

        box_index = tf.expand_dims(tf.range(0, batch_size), 1)
        box_index = tf.tile(box_index, (1, num_rois))
        box_index = tf.reshape(box_index, [-1])

        rs = tf.image.crop_and_resize(img, tf.reshape(rois, [-1, 4]), box_index, (pool_size, pool_size))

        final_output = K.reshape(rs, (batch_size, num_rois, pool_size, pool_size, nb_channels))
        return final_output
