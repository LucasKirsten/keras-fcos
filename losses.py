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

''' Probabilistic IoU '''

EPS = 1e-3

def helinger_dist(x1,y1,a1,b1,c1, x2,y2,a2,b2,c2, freezed=False):
    
    B1 = 1/4.*( (a1+a2)*(y1-y2)**2. + (b1+b2)*(x1-x2)**2. )/( (a1+a2)*(b1+b2) - (c1+c2)**2. + EPS )
    if freezed:
        B2 = 0.
    else:
        sqrt = (a1*b1-c1**2)*(a2*b2-c2**2)
        sqrt = tf.where(sqrt<0, EPS, sqrt)
        B2 = ( (a1+a2)*(b1+b2) - (c1+c2)**2. )/( 4.*tf.math.sqrt(sqrt) + EPS )
        B2 = tf.where(B2<0, EPS, B2)
        B2 = 1/2.*tf.math.log(B2 + EPS)
    
    Bd = B1 + B2
        
    Db = tf.clip_by_value(Bd, EPS, 100.)
    
    return tf.math.sqrt(1 - tf.math.exp(-Db) + EPS)

def get_piou_values(points):
    # xmin, ymin, xmax, ymax
    xmin = points[:,0]; ymin = points[:,1]
    xmax = points[:,2]; ymax = points[:,3]
    angles = points[:,-1]
    
    # get ProbIoU values without rotation
    x = (xmin + xmax)/2.
    y = (ymin + ymax)/2.
    a = tf.math.pow((xmax - xmin), 2.)/12.
    b = tf.math.pow((ymax - ymin), 2.)/12.
    
    # convert values to rotations
    a = a*tf.math.pow(tf.math.cos(angles), 2.) + b*tf.math.pow(tf.math.sin(angles), 2.)
    b = a*tf.math.pow(tf.math.sin(angles), 2.) + b*tf.math.pow(tf.math.cos(angles), 2.)
    c = a*tf.math.cos(angles)*tf.math.sin(angles) - b*tf.math.sin(angles)*tf.math.cos(angles)
    return x, y, a, b, c

def calc_piou(mode, regression_target, regression, freezed=False):
    
    l1 = helinger_dist(
                *get_piou_values(regression_target),
                *get_piou_values(regression),
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
        y_regr_true = y_true[:, :5]
        y_centerness_true = y_true[:, 5]
        
        loss = calc_piou(mode, y_regr_true, y_regr_pred, freezed=freeze_iterations>it)
        it += 1
        
        loss = tf.cast(weight, 'float32') * loss
        loss = tf.reduce_sum(loss * y_centerness_true) / (tf.reduce_sum(y_centerness_true) + 1e-8)
        return loss

    return _iou