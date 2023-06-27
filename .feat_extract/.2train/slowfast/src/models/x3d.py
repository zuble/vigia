import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.keras import regularizers

from yacs.config import CfgNode
import math

from src.models.builder import register_model


'''
_TEMPORAL_KERNEL_BASIS = {
    "x3d": [
            [[5]],  # conv1 temporal kernels.
            [[3]],  # res2 temporal kernels.
            [[3]],  # res3 temporal kernels.
            [[3]],  # res4 temporal kernels.
            [[3]],  # res5 temporal kernels.
        ],
}
'''

@register_model("X3D")
class X3D(K.Model):
  '''
  Constructs the X3D model, given the model configurations
  See: https://arxiv.org/abs/2004.04730v1
  '''
  def __init__(self, cfg: CfgNode):
    '''
    Initialize an instance of the model given the model configurations

    Args:
        cfg (CfgNode): the model configurations
    '''
    super(X3D, self).__init__()
    self.num_classes = cfg.MODEL.NUM_CLASSES
    self._bn_cfg = cfg.X3D.BN

    # for handling ensemble predictions
    # used in call() if not
    #self._num_preds = cfg.TEST.NUM_TEMPORAL_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

    # this block deals with the casw where the width expansion factor is not 1.
    # In the paper, a width factor of 1 results in 24 features from the conv_1
    # layer. If X3D.SCALE_RES2 is True, then the width expansion factor is
    # applied to a channel size of 12
    if cfg.X3D.SCALE_RES2: # apply width_factor directly to channel dimension
      self._conv1_dim = round_width(cfg.X3D.DIM_C1,
          cfg.X3D.WIDTH_FACTOR)
      self._multiplier = 1 # increase the width by 2 for the other blocks
    else: # increase the number of channels by a factor of 2
      self._conv1_dim = round_width(cfg.X3D.DIM_C1, 2)
      self._multiplier = 2

    # [depth, num_channels] for each residual stage
    self._block_basis = [
        [1, cfg.X3D.DIM_C1 * self._multiplier],
        [2, round_width(cfg.X3D.DIM_C1 * self._multiplier, 2)],
        [5, round_width(cfg.X3D.DIM_C1 * self._multiplier, 4)],
        [3, round_width(cfg.X3D.DIM_C1 * self._multiplier, 8)]]

    # regularizer
    l2 = regularizers.l2(cfg.X3D.L2_WEIGHT_DECAY)
    
    # the first layer of the model before the residual stages
    self.conv1 = X3D_Stem(
        regularizer=l2,
        bn_cfg=cfg.X3D.BN,
        out_channels=self._conv1_dim,
        temp_filter_size=cfg.X3D.C1_TEMP_FILTER)

    self.stages = []
    out_dim = self._conv1_dim

    for block in self._block_basis:
        # the output of the previous block
        # is the input of the current block
        in_dim = out_dim

        # apply expansion factors
        out_dim = round_width(block[1], cfg.X3D.WIDTH_FACTOR)
        inner_dim = int(out_dim * cfg.X3D.BOTTLENECK_WIDTH_FACTOR)
        block_depth = round_repeats(block[0], cfg.X3D.DEPTH_FACTOR)

        stage = ResStage(
            in_channels=in_dim,
            inner_channels=inner_dim,
            out_channels=out_dim,
            depth = block_depth,
            bn_cfg=self._bn_cfg,
            regularizer=l2)
        self.stages.append(stage)

    self.conv5 = K.Sequential(name='conv_5')
    self.conv5.add(
        K.layers.Conv3D(
            filters=self.stages[-1]._inner_channels,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            padding='valid',
            use_bias=False,
            data_format='channels_last',
            kernel_regularizer=l2))
    self.conv5.add(
        K.layers.BatchNormalization(
            axis=-1,
            epsilon=self._bn_cfg.EPS,
            momentum=self._bn_cfg.MOMENTUM))
    self.conv5.add(K.layers.Activation('relu'))
    self.pool5 = AdaptiveAvgPool3D(name='pool_5')
    self.fc1 = K.layers.Conv3D(
        filters=2048,
        kernel_size=1,
        strides=1,
        use_bias=False,
        activation='relu',
        name='fc_1',
        kernel_regularizer=l2)
    self.dropout = K.layers.Dropout(rate=cfg.X3D.DROPOUT_RATE)
    self.fc2 = K.layers.Dense(
        units=self.num_classes,
        use_bias=True,
        name='fc_2',
        kernel_regularizer=l2)
    # model output needs to be float32 even if mixed
    # precision policy is set to float16
    self.softmax = K.layers.Softmax(dtype='float32')

    def call(self, input, training=False):
        out = self.conv1(input)
        for stage in self.stages:
            out = stage(out, training=training)
        out = self.conv5(out, training=training)
        out = self.pool5(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        #if not training:
        #  # average predictions
        #  out = tf.reshape(out, (-1, self._num_preds, 1, 1, 1, out.shape[-1]))
        #  out = tf.reduce_mean(out, 1)
        return tf.reshape(out, (-1, self.num_classes))

    def summary(self, input_shape):
        x = K.Input(shape=input_shape)
        model = K.Model(inputs=x, outputs=self.call(x), name='X3D')
        return model.summary()


class X3D_Stem(K.layers.Layer):
    '''
    X3D_Stem: the first layer of the X3D network, connecting
    the data layer and the residual stages. Applies channel-wise
    separable convolution.
    '''
    def __init__(self,
                bn_cfg: CfgNode,
                regularizer: regularizers,
                out_channels: int = 24,
                temp_filter_size: int = 5):
        '''
        Args:
            bn_cfg (CfgNode) containing parameters for the batch
                normalization layer
            regularizer (tf.Keras.regularizers): regularization function
            out_channels (int): number of filters to use in
                the convolutional layers
            temp_filter_size (int): the filter size for the
                temporal convolution
        '''
        super(X3D_Stem, self).__init__(name='conv_1')

        self.bn_momentum = bn_cfg.MOMENTUM
        self.bn_eps = bn_cfg.EPS

        # represents spatial padding of size (0, 1, 1)
        self._spatial_paddings = tf.constant([
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0]])

        # represents temporal padding of size
        # (temp_filter_size//2, 0, 0)
        self._temp_paddings = tf.constant([
            [0, 0],
            [temp_filter_size//2, temp_filter_size//2],
            [0, 0],
            [0, 0],
            [0, 0]])

        # spatial convolution
        self.conv_s = K.layers.Conv3D(
            filters=out_channels,
            kernel_size=(1, 3, 3),
            strides=(1, 2, 2),
            use_bias=False,
            data_format='channels_last',
            kernel_regularizer=regularizer)

        ## temporal convolution
        #self.conv_t = K.layers.Conv3D(
        #    filters=out_channels,
        #    kernel_size=(temp_filter_size, 1, 1),
        #    strides=(1, 1, 1),
        #    use_bias=False,
        #    groups=out_channels,
        #    data_format='channels_last',
        #    kernel_regularizer=regularizer)

        # depthwise temporal convolution
        self.depthwise_conv_t = DepthwiseTemporalConv(
            kernel_size=temp_filter_size,
            filters=out_channels,
            strides=(1, 1),
            padding='same',
            data_format='channels_last',
            depthwise_regularizer=regularizer)
        # pointwise temporal convolution
        self.pointwise_conv_t = K.layers.Conv3D(
            filters=out_channels,
            kernel_size=(1, 1, 1),
            strides=(1, 1, 1),
            use_bias=False,
            data_format='channels_last',
            kernel_regularizer=regularizer)
        
        self.bn = K.layers.BatchNormalization(
            axis=-1,
            momentum=self.bn_momentum,
            epsilon=self.bn_eps)
        self.relu = K.layers.Activation('relu')

    def call(self, input, training=False):
        out = tf.pad(input, self._spatial_paddings)
        out = self.conv_s(out)
        
        out = tf.pad(out, self._temp_paddings)
        #out = self.conv_t(out)
        out = self.depthwise_conv_t(out)
        out = self.pointwise_conv_t(out)
        
        out = self.bn(out, training=training)
        out = self.relu(out)

        return out


class Bottleneck(K.layers.Layer):
    '''
    X3D Bottleneck block: 1x1x1, Tx3x3, 1x1x1 with squeeze-excitation
        added at every 2 stages.
    '''
    def __init__(self,
                channels: tuple,
                bn_cfg: CfgNode,
                regularizer: regularizers,
                stride: int = 1,
                block_index: int = 0,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3):
        '''
        Constructs a single X3D bottleneck block.

        Args:
            channels (tuple): number of channels for each layer in the bottleneck
                order: (number of channles in first two layers, number of channels in last layer)
            bn_cfg (CfgNode): holds the parameters for the batch
            regularizer (tf.Keras.regularizers): regularization function
            stride (int, optional): stride in the spatial dimension.
                Defaults to 1
            block_index (int): the index of the current block
            se_ratio (float, optional): the width multiplier for the squeeze-excitation
                operation. Defaults to 0.0625
            temp_kernel_size (int, optional): the filter size for the
                temporal dimension of the 3x3x3 convolutinal layer.
                Defaults to 3.
        '''
        super(Bottleneck, self).__init__()
        self.block_index = block_index
        self._bn_cfg = bn_cfg

        self.a = K.layers.Conv3D(
            filters=channels[0],
            kernel_size=1,
            strides=(1, 1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_last',
            kernel_regularizer=regularizer)
        
        self.bn_a = K.layers.BatchNormalization(
            axis=-1,
            epsilon=self._bn_cfg.EPS,
            momentum=self._bn_cfg.MOMENTUM)
        self.relu = K.layers.ReLU()
        
        self.b = K.layers.Conv3D(
            filters=channels[0],
            kernel_size=(temp_kernel_size, 3, 3),
            strides=(1, stride, stride), # spatial downsampling
            padding='same',
            use_bias=False,
            groups=channels[0], # turns out to be necessary to reduce model params
            data_format='channels_last',
            kernel_regularizer=regularizer)
        self.bn_b = K.layers.BatchNormalization(
            axis=-1,
            epsilon=self._bn_cfg.EPS,
            momentum=self._bn_cfg.MOMENTUM)
        self.swish = K.layers.Activation('swish')

        # Squeeze-and-Excite operation
        if ((self.block_index + 1) % 2 == 0):
            width = round_width(channels[0], se_ratio)
            self.se_pool = AdaptiveAvgPool3D((1, 1, 1))
            self.se_fc1 = K.layers.Conv3D(
                filters=width,
                kernel_size=1,
                strides=1,
                use_bias=True,
                activation='relu')
            self.se_fc2 = K.layers.Conv3D(
                filters=channels[0],
                kernel_size=1,
                strides=1,
                use_bias=True,
                activation='sigmoid',
                kernel_regularizer=regularizer)

        self.c = K.layers.Conv3D(
            filters=channels[1],
            kernel_size=1,
            strides=(1, 1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_last',
            kernel_regularizer=regularizer)
        self.bn_c = K.layers.BatchNormalization(
            axis=-1,
            epsilon=self._bn_cfg.EPS,
            momentum=self._bn_cfg.MOMENTUM)

    def call(self, input, training=False):
        out = self.a(input)
        out = self.bn_a(out, training=training)
        out = self.relu(out)
        out = self.b(out)
        out = self.bn_b(out, training=training)
        if ((self.block_index + 1) % 2 == 0):
            se = self.se_pool(out)
            se = self.se_fc1(se)
            se = self.se_fc2(se)
            out = out * se
        out = self.swish(out)
        out = self.c(out)
        out = self.bn_c(out, training=training)

        return out


class ResBlock(K.layers.Layer):
    '''
    X3D residual stage: a single residual block
    '''
    _block_index = 0
    def __init__(self,
                channels: tuple,
                bn_cfg: CfgNode,
                regularizer: regularizers,
                stride: int = 1,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3) -> None:
        '''
        Constructs a single X3D residual block.

        Args:
            channels (tuple): (input_channels, inner_channels, output_channels)
            bn_cfg (CfgNode) containing parameters for the batch
                normalization layer
            regularizer (tf.Keras.regularizers): regularization function
            stride (int, optional): stride in the spatial dimension.
                Defaults to 1
            se_ratio (float, optional): the width multiplier for the squeeze-excitation
                operation. Defaults to 0.0625
            temp_kernel_size (int, optional): the filter size for the
                temporal dimension of the 3x3x3 convolutinal layer.
                Defaults to 3.
        '''
        super(ResBlock, self).__init__(name='ResBlock_%u' %ResBlock._block_index)
        ResBlock._block_index += 1

        self.in_channels = channels[0]
        self.inner_channels = channels[1]
        self.out_channels = channels[2]
        self._bn_cfg = bn_cfg

        # handles residual connection after downsampling
        if (self.in_channels != self.out_channels or stride != 1):
            self.residual = K.layers.Conv3D(
                filters=self.out_channels,
                kernel_size=(1, 1, 1),
                strides=(1, stride, stride),
                padding='valid',
                use_bias=False,
                data_format='channels_last',
                kernel_regularizer=regularizer)
            self.bn_r = K.layers.BatchNormalization(
                axis=-1,
                epsilon=self._bn_cfg.EPS,
                momentum=self._bn_cfg.MOMENTUM)

        self.bottleneck = Bottleneck(
            channels=channels[1:],
            stride=stride,
            bn_cfg=self._bn_cfg,
            regularizer=regularizer,
            block_index=ResBlock._block_index,
            se_ratio=se_ratio,
            temp_kernel_size=temp_kernel_size)
        self.add_op = K.layers.Add()
        self.relu = K.layers.Activation('relu')

    def call(self, input, training=False):
        out = self.bottleneck(input)
        if hasattr(self, 'residual'):
            res = self.residual(input)
            res = self.bn_r(res, training=training)
            out = self.add_op([res, out])
        else:
            out = self.add_op([input, out])
        out = self.relu(out)

        return out


class ResStage(K.layers.Layer):
    '''
    Constructs a residual stage of given depth
        for the X3D network
    '''
    _stage_index = 2 # following the convention in the paper
    def __init__(self,
                in_channels: int,
                inner_channels: int,
                out_channels: int,
                depth: int,
                bn_cfg: CfgNode,
                regularizer: regularizers,
                se_ratio: float = 0.0625,
                temp_kernel_size: int = 3):
        '''
        Args:
            in_channels (int): number of channels at the input of
                a stage
            inner_channels (int): the number of channels at the bottleneck
                layers of the stage
            out_channels (int): the number of channels at the output of the
                stage
            depth (int): the depth of the stage or number of times stage
                is repeated
            bn_cfg (CfgNode) containing parameters for the batch
                normalization layer
            regularizer (tf.Keras.regularizers): regularization function
            se_ratio (float, optional): the width multiplier for the squeeze-excitation
                operation. Defaults to 0.0625
            temp_kernel_size (int, optional): the filter size for the
                temporal dimension of the 3x3x3 convolutinal layer.
                Defaults to 3.
        '''
        super(ResStage, self).__init__(name='res_stage_%u' %ResStage._stage_index)
        ResStage._stage_index += 1

        self._bn_cfg = bn_cfg
        self.stage = K.Sequential()
        self._inner_channels = inner_channels

        # the second layer of the first block of each stage
        # does spatial downsampling with a stride of 2
        # the input for subsequent layers is the output
        # from the preceeding layer
        for i in range(depth):
            self.stage.add(
                ResBlock(
                    bn_cfg=self._bn_cfg,
                    se_ratio=se_ratio,
                    regularizer=regularizer,
                    temp_kernel_size=temp_kernel_size,
                    stride=2 if i == 0 else 1,
                    channels=(
                    in_channels if i == 0 else out_channels,
                    inner_channels,
                    out_channels)))

    def call(self, input, training=False):
        return self.stage(input, training=training)


class AdaptiveAvgPool3D(K.layers.Layer):
    '''
    Implementation of AdaptiveAvgPool3D as used in pyTorch impl.
    '''
    def __init__(self,
                spatial_out_shape=(1, 1, 1),
                data_format='channels_last',
                **kwargs) -> None:
        super(AdaptiveAvgPool3D, self).__init__(**kwargs)
        assert len(spatial_out_shape) == 3, "Please specify 3D shape"
        assert data_format in ('channels_last', 'channels_first')

        self.data_format = data_format
        self.out_shape = spatial_out_shape
        self.avg_pool = K.layers.GlobalAveragePooling3D()

    def call(self, input):
        out = self.avg_pool(input)
        if self.data_format == 'channels_last':
            return tf.reshape(
                out,
                shape=(
                    -1,
                    self.out_shape[0],
                    self.out_shape[1],
                    self.out_shape[2],
                    out.shape[1]))
        else:
            return tf.reshape(
                out,
                shape=(
                    -1,
                    out.shape[1],
                    self.out_shape[0],
                    self.out_shape[1],
                    self.out_shape[2]))


class DepthwiseTemporalConv(K.layers.Layer):
    def __init__(self, kernel_size, filters, strides, padding, data_format, depthwise_regularizer):
        super(DepthwiseTemporalConv, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.depthwise_regularizer = depthwise_regularizer

    def build(self, input_shape):
        self.depthwise_kernel = self.add_weight(
            name='depthwise_kernel',
            shape=(self.kernel_size, 1, 1, input_shape[-1]),
            initializer='glorot_uniform',
            regularizer=self.depthwise_regularizer,
            trainable=True)

    def call(self, inputs):
        outputs = []
        for i in range(inputs.shape[1]):
            output = K.backend.conv2d(
                inputs[:, i],
                self.depthwise_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format)
            outputs.append(output)
        return K.backend.stack(outputs, axis=1)
    
def round_width(width, multiplier, min_depth=8, divisor=8):
    """
    Round width of filters based on width multiplier
    from: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py

    Args:
        width (int): the channel dimensions of the input.
        multiplier (float): the multiplication factor.
        min_width (int, optional): the minimum width after multiplication.
            Defaults to 8.
        divisor (int, optional): the new width should be dividable by divisor.
            Defaults to 8.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(width + divisor / 2) // divisor * divisor
    )
    if new_filters < 0.9 * width:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    """
    Round number of layers based on depth multiplier.
    Reference: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py
    """
    multiplier = multiplier
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))