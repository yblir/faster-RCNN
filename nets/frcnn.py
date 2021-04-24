from tensorflow.keras.layers import (Conv2D, Dense, Flatten, Input, Reshape,
                                     TimeDistributed)
from tensorflow.keras.models import Model

from nets.resnet import ResNet50, classifier_layers
from nets.RoiPoolingConv import RoiPoolingConv


# 创建建议框网络
# 该网络结果会对先验框进行调整获得建议框
def get_rpn(base_layers, num_anchors):
    '''
    创建rpn网络
    Parameters
    ----------
    base_layers：resnet50输出的特征层（None,38,38,1024）
    num_anchors：先验框框数量，通常为9，即每个网格分配有9个先验框

    Returns
    -------

    '''
    # 利用一个512通道的3x3卷积进行特征整合
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_initializer='normal', name='rpn_conv1')(base_layers)

    # 利用一个1x1卷积调整通道数，获得预测结果
    # rpn_class只预测该先验框是否包含物体
    anchors_class = Conv2D(num_anchors, (1, 1), activation='sigmoid',
                           kernel_initializer='uniform', name='rpn_out_class')(x)
    # 预测每个先验框的变化量，4代表变化量的x,y,w,h
    anchors_offset = Conv2D(num_anchors * 4, (1, 1), activation='linear',
                            kernel_initializer='zero', name='rpn_out_regress')(x)

    anchors_class = Reshape((-1, 1), name="classification")(anchors_class)
    anchors_offset = Reshape((-1, 4), name="regression")(anchors_offset)

    return [anchors_class, anchors_offset]


#   将共享特征层和建议框传入classifier网络
#   该网络结果会对建议框进行调整获得预测框
def get_classifier(base_layers, input_rois, nb_classes=21, pooling_regions=14):
    '''
    Faster-RCNN网络模型
    Parameters
    ----------
    base_layers: resnet50输出的特征层（None,38,38,1024）
    input_rois:
    nb_classes
    pooling_regions

    Returns
    -------

    '''
    # num_rois:一张图片中建议框数量
    # num_rois, 38, 38, 1024 -> num_rois, 14, 14, 2048
    out_roi_pool = RoiPoolingConv()([base_layers, input_rois, pooling_regions])
    # out_roi_pool = RoiPoolingConv(pooling_regions)([base_layers, input_rois])
    # num_rois, 14, 14, 1024 -> num_rois, 1, 1, 2048
    out = classifier_layers(out_roi_pool)
    # TimeDistributed: 对batch_size中的每一个单独做处理
    # num_rois, 1, 1, 1024 -> num_rois, 2048
    out = TimeDistributed(Flatten())(out)

    # num_rois, 1, 1, 1024 -> num_rois, nb_classes
    # (batch_size,num_rois,nb_classes),None:batch_size,num_rois:每张图片的建议框数量,nb_classes:每个建议框预测的类别
    proposal_boxes_class = TimeDistributed(Dense(nb_classes,
                                                 activation='softmax',
                                                 kernel_initializer='zero'),
                                           name=f'dense_class_{nb_classes}')(out)
    # num_rois, 1, 1, 1024 -> num_rois, 4 * (nb_classes-1)
    # (batch_size,num_rois,4*(nb_classes-1)), 4*(nb_classes-1):每个建议框预测的所有类别的建议框变化量. 这个变化量+建议框=预测框
    proposal_boxes_offset = TimeDistributed(Dense(4 * (nb_classes - 1),
                                                  activation='linear',
                                                  kernel_initializer='zero'),
                                            name=f'dense_regress_{nb_classes}')(out)
    return [proposal_boxes_class, proposal_boxes_offset]


def get_model(config, num_classes):
    '''
    创建训练网络模型
    Parameters
    ----------
    config
    num_classes

    Returns
    -------

    '''
    # 输入主干提取的图片
    inputs = Input(shape=(None, None, 3))
    # roi-pooling层的输入，从rpn层获得的最后建议框，None:一张图片中建议框数量
    roi_input = Input(shape=(None, 4))

    #   假设输入为600,600,3
    #   获得一个38,38,1024的共享特征层base_layers
    base_layers = ResNet50(inputs)

    #   每个特征点9个先验框，先验框边长数量*先验框宽高比例数
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    #   将共享特征层传入建议框网络
    #   该网络结果会对先验框进行调整获得建议框
    rpn = get_rpn(base_layers, num_anchors)
    # 这是包含在下面moel_all中的子网络，不必单独训练model_rpn,因为下面model_all训练时会同步更新
    # model_rpn的权重。model_rpn的作用是每次训练和预测时生成用于截取特征图的建议框。训练时每个batch_size
    # 都会更新权重，向好的方向调整，获得的建议框也会越来越精确。额~，我怎么突然想起了GAN网络...
    model_rpn = Model(inputs, rpn)

    #   将共享特征层和建议框传入classifier网络
    #   该网络结果会对建议框进行调整获得预测框
    classifier = get_classifier(base_layers, roi_input, num_classes, config.pooling_regions)
    # 构建包含rpn和分类的两个网络，一起训练，使得两个网络的损失函数整体最小
    model_all = Model([inputs, roi_input], rpn + classifier)

    return model_rpn, model_all


def get_predict_model(config, num_classes):
    '''
    训练时两步一起训练,使得loss全部最小,预测时分开,加快预测速度
    Parameters
    ----------
    config
    num_classes

    Returns
    -------

    '''
    # 输入主干提取的图片
    inputs = Input(shape=(None, None, 3))
    # roi-pooling层的输入，从rpn层获得的最后建议框，None:一张图片中建议框数量
    roi_input = Input(shape=(None, 4))
    # 主干网络输出的特征层，预测时作为分类网络的输入
    feature_map_input = Input(shape=(None, None, 1024))

    # 假设输入为600,600,3, 获得一个38,38,1024的共享特征层base_layers
    base_layers = ResNet50(inputs)

    #   每个特征点9个先验框
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)

    #   将共享特征层传入建议框网络
    #   该网络结果会对先验框进行调整获得建议框
    rpn = get_rpn(base_layers, num_anchors)
    model_rpn = Model(inputs, rpn + [base_layers])

    #   将共享特征层和建议框传入classifier网络
    #   该网络结果会对建议框进行调整获得预测框
    classifier = get_classifier(feature_map_input, roi_input, num_classes, config.pooling_regions)
    # 此处仅构建分类模型，与训练时的模型不同
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    return model_rpn, model_classifier_only
