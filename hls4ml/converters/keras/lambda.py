from hls4ml.converters.keras_to_hls import keras_handler


# implementation only for channel_last case (layer['n_chan'] = inp0[-1])
# because keras_layer['config'] does not have 'data_format' key.
@keras_handler('Lambda')
def parse_lambda_layer(keras_layer, input_names, input_shapes, data_reader):

    # layer = parse_default_keras_layer(keras_layer, input_names)
    layer = {}
    layer['name'] = keras_layer['config']['name']
    layer['inputs'] = [input_names[0]]
    layer['class_name'] = 'Lambda'

    (layer['in_height'], layer['in_width'], layer['n_chan']) = input_shapes[0][-3], input_shapes[0][-2], input_shapes[0][-1]
    (layer['out_height'], layer['out_width'], layer['n_filt']) = (layer['in_height'], layer['in_width'], layer['n_chan'])

    layer['depth_to_space'], layer['normalize'], layer['denormalize'] = 0, 0, 0
    if 'denormalize' in layer['name']:
        layer['denormalize'] = 1
    elif 'normalize' in layer['name']:
        layer['normalize'] = 1
    elif 'depth_to_space' in layer['name']:
        layer['depth_to_space'] = 1
        bs = keras_layer['config']['arguments']['block_size']
        layer['block_size'] = bs
        layer['n_filt'] //= bs**2
        layer['out_height'] *= bs
        layer['out_width'] *= bs
    else:
        raise Exception('Unknown lambda layer name: {}'.format(layer['name']))

    output_shape = [input_shapes[0][0], layer['out_height'], layer['out_width'], layer['n_filt']]

    return layer, output_shape
