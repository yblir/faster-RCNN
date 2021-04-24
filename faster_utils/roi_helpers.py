import numpy as np


def bbox_iou(bbox_a, bbox_b):
    '''
    计算建议框和真实框的重合程度
    Parameters
    ----------
    bbox_a
    bbox_b

    Returns
    -------

    '''
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox2loc(pbox, tbox):
    '''
    计算真实框与建议框的偏移量,这里与rpn网络预测的处理手法一样,都是
    求变化量.rpn网格预测的变化量+先验框=建议框.
    分类网络预测的偏移量+建议框=网络最终预测框
    Parameters
    ----------
    pbox:已提取的正负样本的坐标,[x_min,y_min,x_max,y_max]
    tbox:正负样本对应的真实框的坐标,[x_min,y_min,x_max,y_max]

    Returns
    -------

    '''
    # 获得建议框宽高与中心
    proposal_w = pbox[:, 2] - pbox[:, 0]
    proposal_h = pbox[:, 3] - pbox[:, 1]
    proposal_center_x = pbox[:, 0] + 0.5 * proposal_w
    proposal_center_y = pbox[:, 1] + 0.5 * proposal_h

    # 获得真实框宽高与中心
    true_w = tbox[:, 2] - tbox[:, 0]
    true_h = tbox[:, 3] - tbox[:, 1]
    true_center_x = tbox[:, 0] + 0.5 * true_w
    true_center_y = tbox[:, 1] + 0.5 * true_h

    eps = np.finfo(proposal_h.dtype).eps
    # eps=0.0001
    proposal_w = np.maximum(proposal_w, eps)
    proposal_h = np.maximum(proposal_h, eps)

    # 计算真实框与建议框偏移量并归一化
    dx = (true_center_x - proposal_center_x) / proposal_w
    dy = (true_center_y - proposal_center_y) / proposal_h
    dw = np.log(true_w / proposal_w)
    dh = np.log(true_h / proposal_h)

    # loc.shape=(n,4),n:正负样本总数,值为建议框编号,4:真实框与建议框中心点偏差量和框高比例值
    # 并且所有值都已经归一化
    loc = np.concatenate([dx[:, None], dy[:, None],
                          dw[:, None], dh[:, None]], axis=-1)
    return loc


def calc_iou(proposal_boxes, config, true_boxes, num_classes):
    true_boxes_loc = true_boxes[:, :4]
    true_boxes_cls = true_boxes[:, 4]

    if len(true_boxes_loc) == 0:
        # 若某张图片不存在真实框(标注框),则将真实框坐标,真实框与建议框iou值,真实框标签全部置为0
        true_box_idx = np.zeros(len(proposal_boxes), np.int32)
        max_iou = np.zeros(len(proposal_boxes))
        gt_roi_label = np.zeros(len(proposal_boxes))
    else:
        # 计算建议框和真实框的重合程度,iou.shape=(n,x),n:建议框数量,值表示建议框编号, x:真实框数量
        # 即每个建议框与所有真实框的iou值
        iou = bbox_iou(proposal_boxes, true_boxes_loc)

        # 获得每一个建议框最对应的真实框的iou  (n,),n:建议框数量,值表示某个建议框与所有真实框iou中最大的值
        max_iou = iou.max(axis=1)
        # 获得每一个建议框最对应的真实框  (n,),n:建议框数量, 值表示某个建议框与所有真实框iou中最大的值的索引
        # 即获得某个建议框与哪个真实框iou最大,该建议框负责预测这个真实框,预测物体的类别是这个真实框的物体类别
        true_box_idx = iou.argmax(axis=1)
        # 取得建议框应该预测的真实类别 (n,),n:建议框数量, 值表示某个建议框应该预测的类别值
        gt_roi_label = true_boxes_cls[true_box_idx]

    #   满足建议框和真实框iou大于neg_iou_thresh_high的作为负样本
    #   将正样本的数量限制在self.pos_roi_per_image以内

    # 将满足iou条件的建议框设为正样本
    pos_index = np.where(max_iou >= config.classifier_max_iou)[0]
    # 限制一张图片中的正样本数量不超过总样本数的一半
    pos_nums = int(min(config.num_rois // 2, pos_index.size))
    if pos_index.size > 0:
        # 随机从正样本的索引中不重复提取pos_nums个正样本,样本次序会被打乱
        pos_index = np.random.choice(pos_index, size=pos_nums, replace=False)

    #   满足建议框和真实框重合程度小于neg_iou_thresh_high大于neg_iou_thresh_low作为负样本
    #   将正样本的数量和负样本的数量的总和固定成self.n_sample
    neg_index = np.where((max_iou < config.classifier_max_iou) &
                         (max_iou >= config.classifier_min_iou))[0]
    # 一张图片中,样本总数-正样本=负样本数量
    neg_nums = config.num_rois - pos_nums
    if neg_nums > neg_index.size:
        # 当需要的负样本数量多于已存在的负样本,可以重复抽样,不然负样本数量不够呀~
        neg_index = np.random.choice(neg_index, size=neg_nums, replace=True)
    else:
        neg_index = np.random.choice(neg_index, size=neg_nums, replace=False)

    keep_index = np.append(pos_index, neg_index)
    # 提取训练用的所有正阳本与负样本,前半部分为正样本,后半部分为负样本
    pos_neg_boxes = proposal_boxes[keep_index]

    if len(true_boxes_loc) != 0:
        # (n,4),n:正负样本总数,值为建议框编号,4:x,y,w,h.真实框与建议框中心点偏差量和框高比例值
        loc_offset = bbox2loc(pos_neg_boxes, true_boxes_loc[true_box_idx[keep_index]])
        # 预测的应该的变化量*分类系数? x*8,y*8,w*4,h*4,可能会使预测效果更好吧!
        loc_offset = loc_offset * np.array(config.classifier_regr_std)
    else:
        loc_offset = np.zeros_like(pos_neg_boxes)

    # 取得所有正负样本应该预测的类别值
    gt_roi_label = gt_roi_label[keep_index]
    # voc数据集20个类别,包含背景共21个. 0,1,2,...,20. 其中20为背景,
    # 此处将负样本类别全部设为背景 20
    gt_roi_label[pos_nums:] = num_classes - 1

    # 正负样本对应的建议框坐标
    pos_neg_loc = np.zeros_like(pos_neg_boxes)
    # (n,4),n:正负样本总和,前半部分为正样本. x,y,w,h=>y,x,h,w
    pos_neg_loc[:, [0, 1, 2, 3]] = pos_neg_boxes[:, [1, 0, 3, 2]]

    # (n,21),n:建议框数量,21:每个建议框,应该有的预测类别one_hot编码
    label_one_hot = np.eye(num_classes)[np.array(gt_roi_label, np.int32)]

    # 正负样本数总和
    pos_neg_nums = np.shape(loc_offset)[0]
    true_class_label = np.zeros([pos_neg_nums, num_classes - 1, 4])
    true_boxes_offset = np.zeros([pos_neg_nums, num_classes - 1, 4])

    # 将每个正样本对应的真实标签置为1,并把偏移量写入对应位置
    true_class_label[np.arange(pos_nums), np.array(gt_roi_label[:pos_nums], np.int32)] = 1
    true_boxes_offset[np.arange(pos_nums), np.array(gt_roi_label[:pos_nums], np.int32)] = loc_offset[:pos_nums]

    true_class_label = np.reshape(true_class_label, [pos_neg_nums, -1])
    true_boxes_offset = np.reshape(true_boxes_offset, [pos_neg_nums, -1])

    # (n,160): n:建议框数量,160:前80是类别,后80是类别对应的坐标
    true_boxes_offset = np.concatenate([np.array(true_class_label),
                                        np.array(true_boxes_offset)], axis=1)

    return pos_neg_loc, label_one_hot, true_boxes_offset
