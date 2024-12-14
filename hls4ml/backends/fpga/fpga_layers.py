import numpy as np

from hls4ml.model.attributes import Attribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.layers import Conv1D, Conv2D, Lambda, Layer
from hls4ml.model.types import IntegerPrecisionType, XnorPrecisionType


class BatchNormalizationQuantizedTanh(Layer):
    '''Merged Batch Normalization and quantized (binary or ternary) Tanh layer.
    The mean, variance, beta, gamma parameters are folded into the threshold(s) at which the
    sign of the input flips after the quantized (binary or ternary) Tanh activation.
    '''

    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_filt', default=0),
        TypeAttribute('accum'),
        ConfigurableAttribute('reuse_factor', default=1),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, dims, precision=XnorPrecisionType())
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, dims, precision=IntegerPrecisionType(width=2))
        else:
            raise Exception(
                'Unsupported quantize attribute for BatchNormalizationQuantizedTanh: {}'.format(self.get_attr('quantize'))
            )

    def set_thresholds(self, scale, bias, ternary_threshold=0.5):
        inp = self.get_input_variable()
        shape = inp.shape
        dims = inp.dim_names
        precision = self.model.config.backend.convert_precision_string(inp.type.precision)
        F = precision.fractional
        threshold = -bias / scale
        if self.get_attr('quantize') == 2:
            self.add_output_variable(shape, dims, precision=XnorPrecisionType())
            threshold = np.floor(threshold * 2**F) / 2**F
            self.add_weights_variable(
                name='threshold',
                var_name='t{index}',
                data=threshold,
                type_name='threshold{index}_t',
                precision=inp.type.precision,
            )
        elif self.get_attr('quantize') == 3:
            self.add_output_variable(shape, dims, precision=IntegerPrecisionType(width=2))
            threshold_hi = ternary_threshold / scale + threshold
            threshold_lo = -ternary_threshold / scale + threshold
            threshold_hi = np.floor(threshold_hi * 2**F) / 2**F
            threshold_lo = np.floor(threshold_lo * 2**F) / 2**F
            self.add_weights_variable(
                name='threshold_hi',
                var_name='th{index}',
                data=threshold_hi,
                type_name='threshold_hi_{index}_t',
                precision=inp.type.precision,
            )
            self.add_weights_variable(
                name='threshold_lo',
                var_name='tl{index}',
                data=threshold_lo,
                type_name='threshold_lo_{index}_t',
                precision=inp.type.precision,
            )


class PointwiseConv1D(Conv1D):
    '''Optimized Conv1D implementation for 1x1 kernels.'''

    # Nothing to do, will pick up function and config from class name
    pass


class PointwiseConv2D(Conv2D):
    '''Optimized Conv2D implementation for 1x1 kernels.'''

    # Nothing to do, will pick up function and config from class name
    pass


class DepthToSpace(Lambda):
    '''tensorflow.nn.depth_to_space implementation'''

    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_chan'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_filt'),
        Attribute('block_size'),
        Attribute('depth_to_space'),
        Attribute('normalize'),
        Attribute('denormalize'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = list(inp.shape)
        bs = self.get_attr('block_size')
        shape[-1] //= bs**2
        shape[-2] *= bs
        shape[-3] *= bs
        dims = [f'OUT_HEIGHT_{self.index}', f'OUT_WIDTH_{self.index}', f'N_FILT_{self.index}']
        self.add_output_variable(shape, dims)


class Normalize(Lambda):
    '''Normalization implementation of the pixel values (x - rgb_mean) / 127.5'''

    # Nothing to do, will pick up function and config from class name
    pass


class Denormalize(Lambda):
    '''Denormalization implementation of the pixel values (x * 127.5 + rgb_mean)'''

    # Nothing to do, will pick up function and config from class name
    pass
