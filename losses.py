"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import tensorflow as tf
import keras.backend as K

def focal(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # compute the focal loss
        location_state = y_true[:, :, -1]
        labels = y_true[:, :, :-1]
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(K.equal(labels, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * K.binary_crossentropy(labels, y_pred)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(location_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(cls_loss) / normalizer

    return _focal


def iou():
    def iou_(y_true, y_pred):
        location_state = y_true[:, :, -1]
        indices = tf.where(K.equal(location_state, 1))
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_regr_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_regr_true = y_true[:, :4]
        y_centerness_true = y_true[:, 4]

        # (num_pos, )
        pred_left = y_regr_pred[:, 0]
        pred_top = y_regr_pred[:, 1]
        pred_right = y_regr_pred[:, 2]
        pred_bottom = y_regr_pred[:, 3]

        # (num_pos, )
        target_left = y_regr_true[:, 0]
        target_top = y_regr_true[:, 1]
        target_right = y_regr_true[:, 2]
        target_bottom = y_regr_true[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # (num_pos, )
        intersection_over_union = (area_intersect + 1.0) / (area_union + 1.0)
        losses = -tf.log(intersection_over_union)
        losses = tf.reduce_sum(losses * y_centerness_true) / (tf.reduce_sum(y_centerness_true) + 1e-8)
        return losses

    return iou_


def bce():
    def bce_(y_true, y_pred):
        location_state = y_true[:, :, -1]
        indices = tf.where(K.equal(location_state, 1))
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_centerness_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_centerness_true = y_true[:, 0:1]
        loss = K.switch(tf.size(y_centerness_true) > 0,
                        K.binary_crossentropy(target=y_centerness_true, output=y_centerness_pred),
                        tf.constant(0.0))
        loss = K.mean(loss)
        return loss

    return bce_

""" REGRESSION LOSSES """

EPS = 1e-3

def helinger_dist(x1,y1,a1,b1, x2,y2,a2,b2, freezed=False):
    '''
    Dh = sqrt(1 - exp(-Db))
    
    Db = 1/4*((x1-x2)²/(a1+a2) + (y1-y2)²/(b1+b2))-ln2 \
    1/2*ln((a1+a2)*(b1+b2)) - 1/4*ln(a1*a2*b1*b2)
    '''
    
    if freezed:
        B1 = 1/4.*(tf.math.pow(x1-x2, 2.)/(a1+a2+EPS) + tf.math.pow(y1-y2, 2.)/(b1+b2+EPS))
        B2 = 1/2.*tf.math.log((a1+a2)*(b1+b2)+EPS)
        B3 = 1/4.*tf.math.log(a1*a2*b1*b2+EPS)
        Db = B1 + B2 - B3 - tf.math.log(2.)
    else:
        Db = tf.math.pow(x1-x2, 2.)/(2*a1+EPS) + tf.math.pow(y1-y2, 2.)/(2*b1+EPS)
        
    Db = tf.clip_by_value(Db, EPS, 100.)
    
    return tf.math.sqrt(1 - tf.math.exp(-Db) + EPS)

def get_piou_values(array):
    # xmin, ymin, xmax, ymax
    xmin = array[:,0]; ymin = array[:,1]
    xmax = array[:,2]; ymax = array[:,3]
    
    # get ProbIoU values
    x = (xmin + xmax)/2.
    y = (ymin + ymax)/2.
    a = tf.math.pow((xmax - xmin), 2.)/12.
    b = tf.math.pow((ymax - ymin), 2.)/12.
    return x, y, a, b

def calc_piou(mode, target, pred, freezed=False):
    
    l1 = helinger_dist(
                *get_piou_values(target),
                *get_piou_values(pred),
                freezed=freezed
            )
    if mode=='piou_l1':
        return l1
    
    l2 = tf.math.pow(l1, 2.)
    if mode=='piou_l2':
        return l2
    
    l3 = - tf.math.log(1. - l2 + EPS)
    if mode=='piou_l3':
        return l3
    
def calc_diou_ciou(mode, bboxes1, bboxes2):
    # xmin, ymin, xmax, ymax
    
    rows = tf.cast(tf.shape(bboxes1)[0], 'float32')
    cols = tf.cast(tf.shape(bboxes2)[0], 'float32')
    
    if rows * cols == 0:
        return cious

    def cond_true(bboxes1, bboxes2, rows, cols):
        def _return():
            cious = tf.zeros((cols, rows), dtype='float32')
            dious = tf.zeros((cols, rows), dtype='float32')
            exchange = True
            return bboxes2, bboxes1, cious, dious, exchange
        return _return
        
    def cond_false(bboxes1, bboxes2, rows, cols):
        def _return():
            cious = tf.zeros((rows, cols), dtype='float32')
            dious = tf.zeros((rows, cols), dtype='float32')
            exchange = False
            return bboxes1, bboxes2, cious, dious, exchange
        return _return

    bboxes1, bboxes2, cious, dious, exchange = tf.cond(
        rows > cols,
        cond_true(bboxes1, bboxes2, rows, cols),
        cond_false(bboxes1, bboxes2, rows, cols)
    )

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2.
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2.
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2.
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2.

    inter_max_xy = tf.math.minimum(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = tf.math.maximum(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = tf.math.maximum(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = tf.math.minimum(bboxes1[:, :2],bboxes2[:, :2])
    
    inter = inter_max_xy - inter_min_xy
    inter = tf.clip_by_value(inter, 0, tf.reduce_max(inter)+1)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2. + (center_y2 - center_y1)**2.
    outer = out_max_xy - out_min_xy
    outer = tf.clip_by_value(outer, 0, tf.reduce_max(outer)+1)
    outer_diag = (outer[:, 0] ** 2.) + (outer[:, 1] ** 2.)
    union = area1+area2-inter_area
    
    if mode=='diou':
        dious = inter_area / union - (inter_diag) / outer_diag
        dious = tf.clip_by_value(dious, -1.0, 1.0)
        
        dious = tf.cond(exchange, lambda:tf.transpose(dious), lambda:dious)
        return 1. - dious
    
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    v = (4. / (math.pi ** 2.)) * tf.math.pow((tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1)), 2.)
    
    S = tf.stop_gradient(1. - iou)
    alpha = tf.stop_gradient(v / (S + v))
    
    cious = iou - (u + alpha * v)
    cious = tf.clip_by_value(cious, -1.0, 1.0)
    
    cious = tf.cond(exchange, lambda:tf.transpose(cious), lambda:cious)
    
    return 1. - cious

def calc_iou_giou(mode, y_regr_true, y_regr_pred):

    # (num_pos, )
    pred_left = y_regr_pred[:, 0]
    pred_top = y_regr_pred[:, 1]
    pred_right = y_regr_pred[:, 2]
    pred_bottom = y_regr_pred[:, 3]

    # (num_pos, )
    target_left = y_regr_true[:, 0]
    target_top = y_regr_true[:, 1]
    target_right = y_regr_true[:, 2]
    target_bottom = y_regr_true[:, 3]

    target_area = (target_left + target_right) * (target_top + target_bottom)
    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
    w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
    h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)
    
    if mode == 'giou':
        # smallest enclosing box
        x1c = tf.maximum(pred_left, target_left)
        x2c = tf.minimum(pred_right, target_right)
        y1c = tf.maximum(pred_top, target_top)
        y2c = tf.minimum(pred_bottom, target_bottom)

        Ac = (x2c - x1c) * (y2c - y1c)

    area_intersect = w_intersect * h_intersect
    area_union = target_area + pred_area - area_intersect

    # (num_pos, )
    intersection_over_union = (area_intersect + 1.0) / (area_union + 1.0)
    if mode == 'giou':
        intersection_over_union = intersection_over_union - (Ac - area_union)/Ac
        
    return 1. - intersection_over_union

def iou_loss(mode, weight, freeze_iterations=0):
    it = 0
    
    def _iou(y_true, y_pred):
        nonlocal it
        
        location_state = y_true[:, :, -1]
        indices = tf.where(K.equal(location_state, 1))
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_regr_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_regr_true = y_true[:, :4]
        y_centerness_true = y_true[:, 4]
        
        if 'piou' in mode:
            loss = calc_piou(mode, y_regr_true, y_regr_pred, freezed=freeze_iterations>it)
            it += 1
            
        elif mode in ('diou', 'ciou'):
            loss = calc_diou_ciou(mode, y_regr_pred, y_regr_true)
            
        elif mode in ('iou', 'giou'):
            loss = calc_iou_giou(mode, y_regr_true, y_regr_pred)
        
        loss = tf.cast(weight, 'float32') * loss
        loss = tf.reduce_sum(loss * y_centerness_true) / (tf.reduce_sum(y_centerness_true) + 1e-8)
        return loss

    return _iou