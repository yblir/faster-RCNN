import colorsys
import copy
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model

from nets.frcnn import get_model, get_predict_model

from nets.data_gen import get_new_img_size
from faster_utils.anchors import get_anchors
from faster_utils.config import Config
from faster_utils.utils_new import BBoxUtility


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的NUM_CLASSES、
#   model_path和classes_path参数的修改
# --------------------------------------------#
class FRCNN(object):
    _defaults = {
        "model_path": 'model_data/voc_weights.h5',
        # "model_path": 'logs/Epoch2-Total_Loss509791.9103852113-Val_Loss904541.3541666666.h5',
        "classes_path": 'model_data/voc_classes.txt',
        "confidence": 0.5,
        "iou": 0.3
    }

    #   初始化faster RCNN

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config()
        self.colors = self.set_colors()
        self.bbox_util = BBoxUtility()
        self.model_rpn, self.model_classifier = self.load_rcnn_model()

    def load_rcnn_model(self):
        '''
        加载分开预测的载模型
        Returns
        -------

        '''
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        try:
            # 若是完整的模型,直接载入,不需网络结构,也不需要编译.
            model_rpn = load_model(model_path, compile=False)
            model_classifier = load_model(model_path, compile=False)
        except Exception as v:
            print(v)
            # 包含背景,所以要+1
            num_classes = len(self.class_names) + 1
            try:
                model_rpn, model_classifier = get_predict_model(self.config, num_classes)
                model_rpn.load_weights(self.model_path, by_name=True)
                model_classifier.load_weights(self.model_path, by_name=True)
            except Exception as e:
                print('模型加载失败:', e)
                raise

        return model_rpn, model_classifier

    #   获得所有的分类
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # 为每种类别分配框颜色
    def set_colors(self):
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        # *x: 解包(10,1.,1,.)这样的结构
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # [(12,233,9),(...),(...)]  # 每个小元组就是一个rgb色彩值
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        return colors

    #   用于计算共享特征层的大小
    def get_img_output_length(self, new_w, new_h):
        def get_output_length(input_length):
            # input_length += 6
            filter_sizes = [7, 3, 1, 1]
            padding = [3, 1, 0, 0]
            stride = 2
            for i in range(4):
                # input_length = (input_length - filter_size + stride) // stride
                input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
            return input_length

        return get_output_length(new_w), get_output_length(new_h)

    # 编译为静态图，提高预测效率
    @tf.function(experimental_relax_shapes=True)
    def rpn_model_pred(self, photo):
        preds = self.model_rpn(photo, training=False)
        return preds

    @tf.function(experimental_relax_shapes=True)
    def classifier_model_pred(self, photo):
        preds = self.model_classifier(photo, training=False)
        return preds

    #   检测图片
    def detect_image(self, image):
        img_w, img_h = image.size
        old_image = copy.deepcopy(image)

        #   给原图像进行resize，resize到短边为600的大小上
        new_w, new_h = get_new_img_size(img_w, img_h)
        image = image.resize([new_w, new_h], Image.BICUBIC)
        photo = np.array(image, dtype=np.float64)

        #   图片预处理，归一化。
        photo = preprocess_input(np.expand_dims(photo, 0))
        # 预测值有3个, rpn类别,框变化量, resnet的输出:公共特征层
        rpn_pred = self.rpn_model_pred(photo)
        rpn_pred = [x.numpy() for x in rpn_pred]

        # 将建议框网络的预测结果进行解码
        feat_w, feat_h = self.get_img_output_length(new_w, new_h)
        # 获得归一化后的先验框坐标
        anchors = get_anchors([feat_w, feat_h], new_w, new_h)

        # 获得rpn网络的预测框, 并对rpn预测框根据类别置信度大小进行初步筛选
        # 建议框的小数形式
        rpn_results = self.bbox_util.detection_out_rpn(rpn_pred, anchors)

        #   在获得建议框和共享特征层后，将二者传入classifier中进行预测
        feat_layer = rpn_pred[2]
        # 取出rpn网络输出的所有建议框
        proposal_box = np.array(rpn_results)[:, :, 1:]
        temp_ROIs = np.zeros_like(proposal_box)

        # x_min,y_min,x_max,y_max => y_min,x_min,y_max,x_max
        temp_ROIs[:, :, [0, 1, 2, 3]] = proposal_box[:, :, [1, 0, 3, 2]]

        # classifier_pred = self.model_classifier([feat_layer, temp_ROIs])
        classifier_pred = self.model_classifier([feat_layer, temp_ROIs])

        classifier_pred = [x.numpy() for x in classifier_pred]

        #   利用classifier的预测结果对建议框进行解码，获得预测框
        results = self.bbox_util.detection_out_classifier(classifier_pred,
                                                          proposal_box, self.config, self.confidence)

        if len(results[0]) == 0:
            return old_image

        results = np.array(results[0])
        boxes = results[:, :4]
        top_conf = results[:, 4]
        top_label_indices = results[:, 5]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * img_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * img_h

        # 以下步骤全是画框操作，几乎所有目标检测算法都可复用，且方法固定
        # =======================================================
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = max((np.shape(old_image)[0] + np.shape(old_image)[1]) // img_w * 2, 1)

        image = old_image
        for i, c in enumerate(top_label_indices):
            predicted_class = self.class_names[int(c)]
            score = top_conf[i]

            left, top, right, bottom = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[int(c)])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
