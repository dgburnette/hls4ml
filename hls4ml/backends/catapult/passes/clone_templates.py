""" The clone templates in the fpga backend do not work for Catapult because ......
"""

# from hls4ml.backends.fpga.passes.clone import Clone, CloneFunctionTemplate
# from hls4ml.backends.template import FunctionCallTemplate
from hls4ml.backends.fpga.passes.clone import CloneFunctionTemplate

clone_include_list = ['nnet_utils/nnet_stream.h']


class CloneCatapultFunctionTemplate(CloneFunctionTemplate):
    def __init__(self):
        super().__init__()
        # super().__init__(Clone, include_header=clone_include_list)

    def format(self, node):
        params = self._default_function_params(node)
        for i, _output in enumerate(node.outputs):
            params['output' + str(i + 1)] = node.variables[node.outputs[i]].name

        template = (
            'nnet::clone_stream<{input_t}, {output_t}, {index}>({input}, '
            + ', '.join(['{output' + str(i + 1) + '}' for i in range(len(node.outputs))])
            + ');'
        )

        return template.format(**params)
