"""
    Normalize and Denormalize layers for hls4ml
    Takes as an input one arrays: x
    and computes (x - rgb_mean) / 127.5 for the normalize layer
    and (x * 127.5 + rgb_mean) for the denormalize layer

    The HLS part is in contrib/normalize_mean_layer/normalize_mean.h
"""

from pathlib import Path

import hls4ml
from hls4ml.model.attributes import Attribute


# hls4ml implementations
class Normalize(hls4ml.model.layers.Layer):
    """Normalization implementation of the pixel values (x - rgb_mean) / 127.5"""

    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_chan'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_filt'),
    ]
 
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        self.add_output_variable(shape)


# Templates
normalize_config_template = """struct config{index} : nnet::normalize_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
}};\n"""
normalize_function_template = 'nnet::normalize_mean<{input_t}, {output_t}, {config}>({input}, {output});'
normalize_include_list = ['nnet_utils/normalize.h']


class NormalizeConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(Normalize)
        self.template = normalize_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class NormalizeFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(Normalize, include_header=normalize_include_list)
        self.template = normalize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


# hls4ml implementations
class Denormalize(hls4ml.model.layers.Layer):
    """Denormalization implementation of the pixel values (x * 127.5 + rgb_mean)"""

    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_chan'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_filt'),
    ]
 
    def initialize(self):
        inp = self.get_input_variable(self.inputs[0])
        shape = inp.shape
        self.add_output_variable(shape)


# Templates
denormalize_function_template = 'nnet::denormalize_mean<{input_t}, {output_t}, {config}>({input}, {output});'

class DenormalizeConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(Denormalize)
        self.template = normalize_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class DenormalizeFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(Denormalize, include_header=normalize_include_list)
        self.template = denormalize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def main():
    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer('Normalize', Normalize)
    hls4ml.model.layers.register_layer('Denormalize', Denormalize)

    # Register the optimization passes (if any)
    backend = hls4ml.backends.get_backend('Catapult')

    # Register template passes for the given backend
    backend.register_template(NormalizeConfigTemplate)
    backend.register_template(NormalizeFunctionTemplate)
    backend.register_template(DenormalizeConfigTemplate)
    backend.register_template(DenormalizeFunctionTemplate)

    # Register HLS implementation
    p = Path(__file__).parent / 'normalize.h'
    backend.register_source(p)


if __name__ == '__main__':
    main()
