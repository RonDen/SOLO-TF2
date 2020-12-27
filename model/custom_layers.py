# coding=utf-8
# ================================================================
#
#   Author      : LuoDeng
#   Created date: 2020-12-10
#   Description : 自定义层
#
# ================================================================

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model


METHOD_BICUBIC = 'BICUBIC'
METHOD_NEAREST_NEIGHBOR = 'NEAREST_NEIGHBOR'
METHOD_BILINEAR = 'BILINEAR'
METHOD_AREA = 'AREA'


class Conv2dUnit(Model):
    """
    Basic Convolution Unit
    """
    def __init__(self, filters, kernel_size, strides=1, padding='valid', use_bias=False, bn=True, activation='relu') -> None:
        super(Conv2dUnit, self).__init__()
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            activation='linear'
        )
        self.bn = None
        self.activation = None
        if bn:
            self.bn = layers.BatchNormalization()
        if activation == 'relu':
            self.activation = layers.ReLU()
        elif activation == 'leaky_relu':
            self.activation = layers.LeakyReLU(alpha=0.1)
    
    def __call__(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Conv3x3(object):
    """
    3x3 Convolution with batchnorm, relu activation, same padding
    """
    def __init__(self, filter2, strides, use_dcn):
        super(Conv3x3, self).__init__()
        if use_dcn:
            self.conv2d_unit = None
        else:
            self.conv2d_unit = Conv2dUnit(filter2, 3, strides=strides, padding='same', use_bias=False, bn=False, activation=None)
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
    
    def __call__(self, x):
        x = self.conv2d_unit(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InstanceNormalization(Layer):
    """
    Instance Normalization, output shape(N, H, W, C)
    """
    def __init__(self, epsilon=1e-9, **kwargs):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        super(InstanceNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma')
        self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta')
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        # 把同一group的元素融合到一起。IN是GN的特例，当num_groups为c时。
        x_reshape = tf.reshape(x, (N, H * W, C))
        mean = tf.reduce_mean(x_reshape, axis=1, keepdims=True)
        t = tf.square(x_reshape - mean)
        variance = tf.reduce_mean(t, axis=1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = self.gamma * outputs + self.beta
        outputs = tf.reshape(outputs, (N, H, W, C))
        return outputs


class GroupNormalizationMy(Layer):
    """
    GroupNormalization, output shape (N, H, W, C)
    """
    def __init__(self, num_groups, epsilon=1e-9, **kwargs):
        super(GroupNormalization, self).__init__()
        self.epsilon = epsilon
        self.num_groups = num_groups

    def build(self, input_shape):
        super(GroupNormalization, self).build(input_shape)
        shape = (input_shape[-1], )
        self.gamma = self.add_weight(shape=shape, initializer='ones', name='gamma')
        self.beta = self.add_weight(shape=shape, initializer='zeros', name='beta')
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        # 把同一group的元素融合到一起。IN是GN的特例，当num_groups为c时。
        x_reshape = tf.reshape(x, (N, -1, self.num_groups))
        mean = tf.reduce_mean(x_reshape, axis=1, keepdims=True)
        t = tf.square(x_reshape - mean)
        variance = tf.reduce_mean(t, axis=1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (x_reshape - mean) / std
        outputs = tf.reshape(outputs, shape=(N, H * W, C))
        outputs = self.gamma * outputs + self.beta
        outputs = tf.reshape(outputs, shape=(N, H, W, C))
        return outputs



class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.
    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    Arguments:
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as input.
    References:
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(
        self,
        groups: int = 2,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer = "zeros",
        gamma_initializer = "ones",
        beta_regularizer = None,
        gamma_regularizer = None,
        beta_constraint = None,
        gamma_constraint = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

class Resize(Model):
    def __init__(self, h, w, method):
        super(Resize, self).__init__()
        self.h = h
        self.w = w
        self.method = method
    def __call__(self, x):
        m = tf.image.ResizeMethod.BILINEAR
        if self.method == METHOD_BILINEAR:
            m = tf.image.ResizeMethod.BILINEAR
        elif self.method == METHOD_BICUBIC:
            m = tf.image.ResizeMethod.BICUBIC
        elif self.method == METHOD_NEAREST_NEIGHBOR:
            m = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        elif self.method == METHOD_AREA:
            m = tf.image.ResizeMethod.AREA
        a = tf.image.resize(x, [self.h, self.w], method=m)
        return a


if __name__ == "__main__":
    data1 = tf.random.uniform(shape=(1, 416, 416, 3), minval=-1, maxval=1)
    conv1 = Conv3x3(64, 2, False)
    conv2 = Conv3x3(32, 2, False)

    gn1 = GroupNormalization(4)
    in1 = InstanceNormalization()

    out1 = conv1(data1)
    out1 = gn1(out1)
    out2 = conv2(out1)
    out2 = in1(out2)
    
    print(out2.shape)
