from copy import copy

from hls4ml.backends.fpga.fpga_layers import Denormalize, DepthToSpace, Normalize
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import register_layer
from hls4ml.model.optimizer import OptimizerPass

# HLS Templates for depth to space
depthtospace_config_template = """struct config{index} : nnet::depth_to_space_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
    static const unsigned block_size = {block_size};
}};\n"""

depthtospace_function_template = 'nnet::depth_to_space<{input_t}, {output_t}, {config}>({input}, {output});'
depthtospace_include_list = ['nnet_utils/nnet_depthtospace_stream.h']


class DepthToSpaceConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(DepthToSpace)
        self.template = depthtospace_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class DepthToSpaceFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(DepthToSpace, include_header=depthtospace_include_list)
        self.template = depthtospace_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


# HLS Templates for Normalize
normalize_config_template = """struct config{index} : nnet::normalize_config {{
    static const unsigned in_height = {in_height};
    static const unsigned in_width = {in_width};
    static const unsigned n_chan = {n_chan};
}};\n"""

normalize_function_template = 'nnet::normalize_mean<{input_t}, {output_t}, {config}>({input}, {output});'
normalize_include_list = ['nnet_utils/nnet_normalize_stream.h']


class NormalizeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Normalize)
        self.template = normalize_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class NormalizeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Normalize, include_header=normalize_include_list)
        self.template = normalize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


# HLS Templates for denormalize
denormalize_function_template = 'nnet::denormalize_mean<{input_t}, {output_t}, {config}>({input}, {output});'
denormalize_include_list = ['nnet_utils/nnet_normalize_stream.h']


class DenormalizeConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(Denormalize)
        self.template = normalize_config_template

    def format(self, node):
        params = self._default_config_params(node)
        return self.template.format(**params)


class DenormalizeFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(Denormalize, include_header=denormalize_include_list)
        self.template = denormalize_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


def register_lambda(backend):
    register_layer('DepthToSpace', DepthToSpace)
    register_layer('Normalize', Normalize)
    register_layer('Denormalize', Denormalize)

    # Register the optimization passes
    backend.register_pass('optimize_lambda', OptimizeLambda)

    # Register template passes for the given backend
    backend.register_template(DepthToSpaceConfigTemplate)
    backend.register_template(DepthToSpaceFunctionTemplate)
    backend.register_template(NormalizeConfigTemplate)
    backend.register_template(NormalizeFunctionTemplate)
    backend.register_template(DenormalizeConfigTemplate)
    backend.register_template(DenormalizeFunctionTemplate)


class OptimizeLambda(OptimizerPass):
    def match(self, node):
        return node.class_name in ('Lambda')

    def transform(self, model, node):
        if node.get_attr('normalize') == 1:
            pw_node = model.make_node('Normalize', node.name, copy(node.attributes), node.inputs.copy())
        elif node.get_attr('denormalize') == 1:
            pw_node = model.make_node('Denormalize', node.name, copy(node.attributes), node.inputs.copy())
        else:
            pw_node = model.make_node('DepthToSpace', node.name, copy(node.attributes), node.inputs.copy())
        # Set strategy to ensure lowercase string is passed to the template
        # if model.config.is_resource_strategy(pw_node):
        #     pw_node.set_attr('strategy', 'resource')
        # else:
        #     pw_node.set_attr('strategy', 'latency')
        model.replace_node(node, pw_node)

        return True
