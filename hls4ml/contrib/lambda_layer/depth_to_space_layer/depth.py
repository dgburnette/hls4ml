"""
    The class and template definitions for depth_to_space layer
    The layer implements tensorflow.nn.depth_to_space function  
    reference: https://github.com/jmitrevs/SR_extension.git

    The HLS part is in contrib/lambda_layer/depth_to_space_layer/depth.h
"""

from pathlib import Path

import hls4ml
from hls4ml.model.attributes import Attribute


# hls4ml implementations
class DepthToSpace(hls4ml.model.layers.Layer):
    """hls4ml implementation of tensorflow.nn.depth_to_space"""

    _expected_attributes = [
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_chan'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('n_filt'),
        Attribute('block_size'),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        shape = list(inp.shape)
        bs = self.get_attr('block_size')
        shape[-1] //= bs**2
        shape[-2] *= bs
        shape[-3] *= bs
        self.add_output_variable(shape)


# Templates
depthtospace_config_template = """struct config{index} : nnet::depth_to_space_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned block_size = {block_size};
}};\n"""
depthtospace_function_template = 'nnet::depth_to_space<{input_t}, {output_t}, {config}>({input}, {output});'
depthtospace_include_list = ['nnet_utils/depth.h']


class DepthToSpaceConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(DepthToSpace)
        self.template = depthtospace_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class DepthToSpaceFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(DepthToSpace, include_header=depthtospace_include_list)
        self.template = depthtospace_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def main():
    # Register the hls4ml's IR layer
    hls4ml.model.layers.register_layer('DepthToSpace', DepthToSpace)

    # Register the optimization passes (if any)
    backend = hls4ml.backends.get_backend('Catapult')

    # Register template passes for the given backend
    backend.register_template(DepthToSpaceConfigTemplate)
    backend.register_template(DepthToSpaceFunctionTemplate)

    # Register HLS implementation
    p = Path(__file__).parent / 'depth.h'
    backend.register_source(p)


if __name__ == '__main__':
    main()
