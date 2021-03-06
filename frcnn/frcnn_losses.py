import keras.backend as K
from keras.objectives import categorical_crossentropy, binary_crossentropy
import tensorflow as tf

N_cls = 256
lambda_rpn_class = 1.0
lambda_rpn_regr = 1.0
epsilon = 1e-4


def rpn_loss_classification(num_anchors):
    def rpn_loss_cls(y_true, y_pred):
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :],
                                                                                              y_true[:, :, :,
                                                                                              num_anchors:])) / K.sum(
            epsilon + y_true[:, :, :, :num_anchors])
    return rpn_loss_cls


def rpn_loss_regression(num_anchors):
    def rpn_loss_reg(y_true, y_pred):
        x = y_true[:, :, :, 4 * num_anchors] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(y_true[:, :, :, 4 * num_anchors:]
                                       * (x_bool * (0.5 * x_abs * x_abs) + (1 - x_bool) * (x_abs - 0.5)))\
               / K.sum(epsilon + y_true[:, :, :, : 4* num_anchors])
    return rpn_loss_reg
