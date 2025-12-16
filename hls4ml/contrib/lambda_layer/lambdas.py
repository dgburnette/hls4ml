"""
    The general Keras Lambda layer
    The supported functions include depth_to_space, normalize and denormalize. 
    To add suppport for a new custom function, a new condition has be added for the function in "parse_lambda_layer" function (after line 36).
    In addition, the required template and C++ files should be added to a new directory.
    Note that for the parsing to work properly, when defining Lambda layer in Keras, the layer should have a name that contains the function name in it
        e.g. Lambda(normalize, name='normalize_1')

    The HLS part for each function implementation is in contrib/lambda_layer/function_name/function.h
"""

import hls4ml
from hls4ml.contrib.lambda_layer.depth_to_space_layer import depth
from hls4ml.contrib.lambda_layer.normalize_mean_layer import normalize


# Parser for converter
def parse_lambda_layer(keras_layer, input_names, input_shapes, data_reader):
    assert 'Lambda' in keras_layer['class_name']

    # layer = parse_default_keras_layer(keras_layer, input_names)
    layer = {}
    layer['name'] = keras_layer['config']['name']
    layer['inputs'] = [input_names[0]]
    (layer['in_height'], layer['in_width'], layer['n_chan']) = input_shapes[0][-3], input_shapes[0][-2], input_shapes[0][-1]
    (layer['out_height'], layer['out_width'], layer['n_filt']) = (layer['in_height'], layer['in_width'], layer['n_chan'])
    
    if 'denormalize' in layer['name']:
        layer['class_name'] = 'Denormalize'
    elif 'normalize' in layer['name']:
        layer['class_name'] = 'Normalize'
    elif 'depth_to_space' in layer['name']: 
        layer['class_name'] = 'DepthToSpace'
        block_size = keras_layer['config']['arguments']['block_size']
        layer['block_size'] = block_size
        layer['n_filt'] //= block_size**2
        layer['out_height'] *= block_size
        layer['out_width'] *= block_size
    else:
        raise Exception('Unknown lambda layer name: {}'.format(layer['name']))
    
    output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]
    
    return layer, output_shape


def main():
    # Register the converter for custom Keras layer
    hls4ml.converters.register_keras_v2_layer_handler('Lambda', parse_lambda_layer) 

    depth.main()
    normalize.main()


if __name__ == '__main__':
    main()
