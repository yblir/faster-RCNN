import math

import numpy as np
import tensorflow as tf


class BBoxUtility(object):
    def __init__(self, iou_threshold=0.7, ignore_threshold=0.3,
                 rpn_pre_boxes=6000, rpn_nms=0.7, classifier_nms=0.3, top_k=300):
        self.iou_threshold = iou_threshold
        self.ignore_threshold = ignore_threshold
        self.rpn_pre_boxes = rpn_pre_boxes

        self.rpn_nms = rpn_nms
        self.classifier_nms = classifier_nms
        self.top_k = top_k

    def box_iou(self, box):
        '''
        计算先验框与真实框的iou交并比
        Parameters
        ----------
        box

        Returns
        -------

        '''
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_min = np.maximum(self.anchors[:, :2], box[:2])
        inter_max = np.minimum(self.anchors[:, 2:4], box[2:4])

        inter_wh = inter_max - inter_min
        inter_wh = np.maximum(inter_wh, 0)
        inter_area = inter_wh[:, 0] * inter_wh[:, 1]

        # 真实框的面积
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        anchors_area = (self.anchors[:, 2] - self.anchors[:, 0]) * \
                       (self.anchors[:, 3] - self.anchors[:, 1])
        # 计算iou
        iou = inter_area / (anchors_area + box_area - inter_area)

        return iou

    def encode_ignore_box(self, box):
        '''
        返回真实框编码后的值,即对应的先验框应该偏移多少才能获得真实框
        还要获得对应的负样本的框,避免样本不均衡
        Parameters
        ----------
        box:真实框信息,一个真实框的信息, shape=(5,) x,y,w,h,c

        Returns
        -------

        '''
        # shape=(22500,),一个真实框与所有先验框计算iou
        iou = self.box_iou(box)
        # shape=(22500,1),提取待忽略框的iou值
        ignored_box_iou = np.zeros((self.num_anchors, 1))

        #   找到处于忽略阈值范围内的先验框,在这两个阈值之间的框忽略
        ignore_mask = (iou > self.ignore_threshold) & (iou < self.iou_threshold)
        # 将这些待忽略框的iou值写入忽略框的对应位置
        ignored_box_iou[:, 0][ignore_mask] = iou[ignore_mask]

        # (22500,5),4:真实框对应的偏移量,1:真实框与当前先验框的iou值
        encoded_box = np.zeros((self.num_anchors, 4 + 1))

        #   找到与每一个真实框重合程度较高的先验框
        best_iou_mask = iou > self.iou_threshold
        # 如果所有的先验框,与某个真实框iou都小于iou_threshold, 那么就将iou最大的先验框索引位置置为True
        # 保证真实框至少有一个对应的先验框
        if not best_iou_mask.any():
            best_iou_mask[iou.argmax()] = True

        # 取出所有重合度高的先验框的坐标,即正样本,positive_anchors.shape=(n,4),n:正样本数量
        positive_anchors = self.anchors[best_iou_mask]

        #   逆向编码，将真实框转化为rpn预测结果的格式, 计算真实框的中心与长宽
        box_xy = (box[:2] + box[2:]) / 2
        box_wh = box[2:] - box[:2]

        #   再计算重合度较高的先验框的中心与长宽
        positive_anchors_xy = (positive_anchors[:, :2] + positive_anchors[:, 2:4]) / 2
        positive_anchors_wh = positive_anchors[:, 2:4] - positive_anchors[:, :2]

        #   逆向求取rpn网络应该有的预测结果,并归一化,x,y,w,h
        encoded_box[:, :2][best_iou_mask] = (box_xy - positive_anchors_xy) / positive_anchors_wh
        encoded_box[:, 2:4][best_iou_mask] = np.log(box_wh / positive_anchors_wh)
        # 将于真实框交并比满足条件的iou值,写入对应的编码框的最后一个位置
        encoded_box[:, -1][best_iou_mask] = iou[best_iou_mask]

        # 将多维数组拉成一维的
        return encoded_box.ravel(), ignored_box_iou.ravel()
        # return encoded_box, ignored_box

    def assign_boxes(self, boxes, anchors):
        '''
        获得真实框对应的变化量,供先验框调整使用
        Parameters
        ----------
        boxes:真实框信息,x,y,w,h,c
        anchors:由一定规则,为每个网格点生成的先验框

        Returns
        -------

        '''
        self.num_anchors = len(anchors)
        self.anchors = anchors

        # 4:的内容为网络应该有的回归预测结果, 1:先验框是否包含物体,默认为背景
        assignment = np.zeros((self.num_anchors, 4 + 1))

        # 默认所有先验框都不包含物体
        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment

        #   对每一个真实框都进行iou计算,寻找哪些先验框是需要被忽略的
        axis_boxes = np.apply_along_axis(self.encode_ignore_box, 1, boxes[:, :4])
        encoded_boxes = np.array([axis_boxes[i, 0] for i in range(len(axis_boxes))])
        ingored_boxes_iou = np.array([axis_boxes[i, 1] for i in range(len(axis_boxes))])
        # 上面逻辑有些复杂,相当于执行下面的操作
        # encoded_boxes, ingored_boxes = list(), list()
        # for box in boxes:
        #     encoded_box, ignored_box = self.encode_ignore_box(box[:4])
        #     encoded_boxes.append(encoded_box)
        #     ingored_boxes.append(ignored_box)
        #
        # encoded_boxes = np.array(encoded_boxes)
        # ingored_boxes = np.array(ingored_boxes)

        #   [num_true_box, num_anchors, 1] 其中1为iou
        ingored_boxes_iou = ingored_boxes_iou.reshape(-1, self.num_anchors, 1)
        ingored_boxes_iou = ingored_boxes_iou[:, :, 0].max(axis=0)
        ignore_iou_mask = ingored_boxes_iou > 0

        # 将待忽略先验框的设为是负样本, 坐标设为0
        assignment[:, 4][ignore_iou_mask] = -1

        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_priors, 4+1]=>(3,22500,5)
        #   4是编码后的结果，1为iou
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)
        # 同一个先验框与一张图片中的所有真实框都有iou值,有几个真实框,就是几个iou值,找出最大的那个.
        # 有多少先验框就有多少个这样的iou值,这里有22500个
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # 提起是第几个真实框与当前先验框iou最大,(22500,)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)

        # 将最大iou值的索引提取出来
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        #   计算一共有多少先验框满足需求
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # 将编码后的框,每个框对应的最大iou的值取出来,填入assignment中
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]

        #   4代表为当前先验框是否包含目标,将与真实框重叠程度满足阈值的先验框认为包含物体
        assignment[:, 4][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, anchors):
        '''
        根据rpn网络输出的偏移量,移动先验框,获得建议框proposal_box
        Parameters
        ----------
        mbox_loc:rpn网络输出的先验框偏移量
        anchors

        Returns
        -------

        '''
        # 获得先验框的宽与高
        anchors_w = anchors[:, 2] - anchors[:, 0]
        anchors_h = anchors[:, 3] - anchors[:, 1]

        # 获得先验框的中心点
        anchors_x = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_y = (anchors[:, 3] + anchors[:, 1]) / 2

        # 真实框距离先验框中心的xy轴偏移情况
        proposal_center_x = mbox_loc[:, 0] * anchors_w / 4 + anchors_x
        proposal_center_y = mbox_loc[:, 1] * anchors_h / 4 + anchors_y

        # 建议框的宽与高的求取
        proposal_w = np.exp(mbox_loc[:, 2] / 4) * anchors_w
        proposal_h = np.exp(mbox_loc[:, 3] / 4) * anchors_h

        # 获取建议框的左上角与右下角
        proposal_xmin = proposal_center_x - 0.5 * proposal_w
        proposal_ymin = proposal_center_y - 0.5 * proposal_h
        proposal_xmax = proposal_center_x + 0.5 * proposal_w
        proposal_ymax = proposal_center_y + 0.5 * proposal_h

        # 建议框的左上角与右下角进行堆叠
        proposal_box = np.concatenate((proposal_xmin[:, None],
                                       proposal_ymin[:, None],
                                       proposal_xmax[:, None],
                                       proposal_ymax[:, None]), axis=-1)
        # 防止超出0与1, 因为经过了归一化
        proposal_box = np.minimum(np.maximum(proposal_box, 0.0), 1.0)
        return proposal_box

    def detection_out_rpn(self, predictions, anchors):
        '''
        
        Parameters
        ----------
        predictions:rpn网络预测结果,包含三组
        anchors:归一化后的先验框, 相对于原图

        Returns
        -------

        '''

        #   获得种类的置信度
        mbox_conf = predictions[0]

        #   mbox_loc是回归预测结果  
        mbox_loc = predictions[1]

        results = list()
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        # 训练阶段len(mbox_loc)为batch_size
        for i in range(len(mbox_loc)):
            #  利用rpn的框变化量移动先验框,获得建议框坐标
            decode_bbox = self.decode_boxes(mbox_loc[i], anchors)

            #   取出先验框内包含物体的概率
            c_confs = mbox_conf[i, :, 0]
            # 对所有rpn网络预测的先验框是否包含物体的概率,由大到小排序, 获得索引
            confs_max_index = np.argsort(c_confs)[::-1]
            # 取出前rpn_pre_boxes个包含物体的概率的索引
            confs_max_index = confs_max_index[:self.rpn_pre_boxes]

            # 根据排序好的索引,取出对应的是否包含物体的概率和对应的建议框坐标
            c_confs = c_confs[confs_max_index]
            decode_bbox = decode_bbox[confs_max_index, :]

            # 根据是否包含物体的概率大小,对建议框进行iou的非极大抑制
            idx = tf.image.non_max_suppression(decode_bbox, c_confs,
                                               self.top_k, iou_threshold=self.rpn_nms).numpy()
            # 取出在非极大抑制中效果较好的内容
            good_boxes = decode_bbox[idx]
            confs = c_confs[idx].reshape((-1, 1))

            c_pred = np.concatenate((confs, good_boxes), axis=-1)
            results.append(c_pred)

        # (batch_size,self.top_k,5)  top_k此时设为300
        return np.array(results)

    def detection_out_classifier(self, predictions, proposal_box, config, confidence):

        #   获得种类的置信度 
        proposal_conf = predictions[0]
        #   proposal_loc是回归预测结果    
        proposal_loc = predictions[1]

        results = []
        # 对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        for i in range(len(proposal_conf)):
            proposal_pred = []
            proposal_box[i, :, 2] = proposal_box[i, :, 2] - proposal_box[i, :, 0]
            proposal_box[i, :, 3] = proposal_box[i, :, 3] - proposal_box[i, :, 1]
            # 循环300次
            for j in range(proposal_conf[i].shape[0]):
                if np.max(proposal_conf[i][j, :-1]) < confidence:
                    continue
                # 最后一个为背景的概率,舍弃不算
                # 找到最大的概率对应的索引,就是类别号
                label = np.argmax(proposal_conf[i][j, :-1])
                score = np.max(proposal_conf[i][j, :-1])

                # x,y是左上角点
                (x, y, w, h) = proposal_box[i, j, :]
                # 取出该类别对应的建议框坐标, 每个类别有4个坐标值,所以每次偏移4
                (tx, ty, tw, th) = proposal_loc[i][j, 4 * label: 4 * (label + 1)]
                tx /= config.classifier_regr_std[0]
                ty /= config.classifier_regr_std[1]
                tw /= config.classifier_regr_std[2]
                th /= config.classifier_regr_std[3]

                # 建议框中心
                cx = x + w / 2.
                cy = y + h / 2.

                # 调整后的建议框中心
                cx1 = tx * w + cx
                cy1 = ty * h + cy
                w1 = math.exp(tw) * w
                h1 = math.exp(th) * h

                # 调整后的建议框4个坐标点
                x1 = cx1 - w1 / 2.
                y1 = cy1 - h1 / 2.
                x2 = cx1 + w1 / 2
                y2 = cy1 + h1 / 2

                proposal_pred.append([x1, y1, x2, y2, score, label])

            num_classes = np.shape(proposal_conf)[-1]
            proposal_pred = np.array(proposal_pred)
            good_boxes = []
            if len(proposal_pred) != 0:
                for c in range(num_classes):
                    mask = proposal_pred[:, -1] == c
                    if len(proposal_pred[mask]) > 0:
                        boxes_to_process = proposal_pred[:, :4][mask]
                        confs_to_process = proposal_pred[:, 4][mask]
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self.top_k,
                                                           iou_threshold=self.classifier_nms).numpy()
                        # 取出在非极大抑制中效果较好的内容
                        good_boxes.extend(proposal_pred[mask][idx])
            results.append(good_boxes)

        return results
