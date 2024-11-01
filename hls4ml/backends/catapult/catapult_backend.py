import os
import sys
from warnings import warn

import numpy as np

from hls4ml.backends import FPGABackend
from hls4ml.backends.catapult.catapult_types import CatapultArrayVariableConverter
from hls4ml.backends.fpga.fpga_types import ACTypeConverter, HLSTypeConverter
from hls4ml.model.attributes import ChoiceAttribute, ConfigurableAttribute, TypeAttribute
from hls4ml.model.flow import register_flow
from hls4ml.model.layers import (
    GRU,
    LSTM,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Embedding,
    GarNet,
    GarNetStack,
    GlobalPooling1D,
    GlobalPooling2D,
    Layer,
    Pooling1D,
    Pooling2D,
    SeparableConv1D,
    SeparableConv2D,
    SimpleRNN,
    Softmax,
)
from hls4ml.model.optimizer import get_backend_passes, layer_optimizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType, NamedType, PackedType
from hls4ml.report import parse_catapult_report
from hls4ml.utils.fixed_point_utils import ceil_log2


class CatapultBackend(FPGABackend):
    def __init__(self):
        super().__init__('Catapult')
        self._register_layer_attributes()
        self._register_flows()

    def _register_layer_attributes(self):
        # Add RNN-specific attributes, recurrent_reuse_factor and static implementation
        rnn_layers = [
            SimpleRNN,
            LSTM,
            GRU,
        ]

        for layer in rnn_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('recurrent_reuse_factor', default=1))
            attrs.append(ConfigurableAttribute('static', value_type=bool, default=True))
            attrs.append(ConfigurableAttribute('table_size', default=1024))
            attrs.append(TypeAttribute('table', default=FixedPrecisionType(18, 8)))
            self.attribute_map[layer] = attrs

        # Add ParallelizationFactor to Conv1D/2D
        pf_layers = [
            Conv1D,
            Conv2D,
        ]

        for layer in pf_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(ConfigurableAttribute('parallelization_factor', default=1))
            self.attribute_map[layer] = attrs

        # Add ConvImplementation to Convolution+Pooling layers
        cnn_layers = [Conv1D, Conv2D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Pooling1D, Pooling2D]

        for layer in cnn_layers:
            attrs = self.attribute_map.get(layer, [])
            # attrs.append(ConfigurableAttribute('conv_implementation', value_type=str, default='LineBuffer'))
            attrs.append(ChoiceAttribute('conv_implementation', choices=['LineBuffer', 'Encoded'], default='LineBuffer'))
            self.attribute_map[layer] = attrs

        sep_conv_layers = [SeparableConv1D, SeparableConv2D]
        for layer in sep_conv_layers:
            attrs = self.attribute_map.get(layer, [])
            attrs.append(TypeAttribute('dw_output', default=FixedPrecisionType(18, 8)))
            self.attribute_map[layer] = attrs

    def _register_flows(self):
        initializers = self._get_layer_initializers()
        init_flow = register_flow('init_layers', initializers, requires=['optimize'], backend=self.name)

        streaming_passes = [
            'catapult:reshape_stream',
            'catapult:clone_output',
            'catapult:insert_zero_padding_before_conv1d',
            'catapult:insert_zero_padding_before_conv2d',
            'catapult:broadcast_stream',
        ]
        streaming_flow = register_flow('streaming', streaming_passes, requires=[init_flow], backend=self.name)

        quantization_passes = [
            'catapult:merge_batch_norm_quantized_tanh',
            'catapult:quantize_dense_output',
            'fuse_consecutive_batch_normalization',
            'catapult:xnor_pooling',
        ]
        quantization_flow = register_flow('quantization', quantization_passes, requires=[init_flow], backend=self.name)

        optimization_passes = [
            'catapult:remove_final_reshape',
            'catapult:optimize_pointwise_conv',
            'catapult:inplace_parallel_reshape',
            'catapult:inplace_stream_flatten',
            'catapult:skip_softmax',
            'catapult:fix_softmax_table_size',
            'catapult:process_fixed_point_quantizer_layer',
            'infer_precision_types',
        ]
        optimization_flow = register_flow('optimize', optimization_passes, requires=[init_flow], backend=self.name)

        catapult_types = [
            'catapult:transform_types',
            'catapult:register_bram_weights',
            'catapult:generate_conv_streaming_instructions',
            'catapult:apply_resource_strategy',
            'catapult:generate_conv_im2col',
            'catapult:apply_winograd_kernel_transformation',
        ]
        catapult_types_flow = register_flow('specific_types', catapult_types, requires=[init_flow], backend=self.name)

        templates = self._get_layer_templates()
        template_flow = register_flow('apply_templates', self._get_layer_templates, requires=[init_flow], backend=self.name)

        writer_passes = ['make_stamp', 'catapult:write_hls']
        self._writer_flow = register_flow('write', writer_passes, requires=['catapult:ip'], backend=self.name)

        fifo_depth_opt_passes = [
            'catapult:fifo_depth_optimization'
        ] + writer_passes  # After optimization, a new project will be written

        register_flow('fifo_depth_optimization', fifo_depth_opt_passes, requires=[self._writer_flow], backend=self.name)

        all_passes = get_backend_passes(self.name)

        extras = [
            # Ideally this should be empty
            opt_pass
            for opt_pass in all_passes
            if opt_pass
            not in initializers
            + streaming_passes
            + quantization_passes
            + optimization_passes
            + catapult_types
            + templates
            + writer_passes
            + fifo_depth_opt_passes
        ]

        if len(extras) > 0:
            for opt in extras:
                warn(f'WARNING: Optimizer "{opt}" is not part of any flow and will not be executed.')

        ip_flow_requirements = [
            'optimize',
            init_flow,
            streaming_flow,
            quantization_flow,
            optimization_flow,
            catapult_types_flow,
            template_flow,
        ]

        self._default_flow = register_flow('ip', None, requires=ip_flow_requirements, backend=self.name)

    def get_default_flow(self):
        return self._default_flow

    def get_writer_flow(self):
        return self._writer_flow

    def create_initial_config(
        self,
        tech='fpga',
        part='xcku115-flvb2104-2-i',
        asiclibs='nangate-45nm',
        asicfifo='hls4ml_lib.mgc_pipe_mem',
        fifo=None,
        clock_period=5,
        io_type='io_parallel',
        namespace=None,
        write_weights_txt=True,
        write_tar=False,
        project_dir=None,
        csim=1,
        SCVerify=1,
        Synth=1,
        vhdl=1,
        verilog=1,
        RTLSynth=0,
        RandomTBFrames=2,
        PowerEst=0,
        PowerOpt=0,
        BuildBUP=0,
        BUPWorkers=0,
        LaunchDA=0, **_
    ):
        """Create initial configuration of the Vivado backend.

        Args:
            tech (str, optional): The target technology type. One of 'asic' or 'fpga'.
            part (str, optional): The FPGA part to be used. Defaults to 'xcvu13p-flga2577-2-e'.
            asiclibs (str, optional): The list of ASIC Catapult libraries to load. Defaults to 'nangate-45nm'.
            asicfifo (str, optional): The name of the ASIC FIFO library module to use. Defaults to 'hls4ml_lib.mgc_pipe_mem'.
            fifo (str, optional): The name of the FPGA FIFO library module to use. Default to None.
            clock_period (int, optional): The clock period. Defaults to 5.
            io_type (str, optional): Type of implementation used. One of
                'io_parallel' or 'io_stream'. Defaults to 'io_parallel'.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            namespace (str, optional): If defined, place all generated code within a namespace. Defaults to None.
            write_weights_txt (bool, optional): If True, writes weights to .txt files which speeds up compilation.
                Defaults to True.
            write_tar (bool, optional): If True, compresses the output directory into a .tar.gz file. Defaults to False.
            project_dir (str, optional): The name for the Catapult project directory. Defaults to None.
            csim (bool, optional): Enables/disables C model simulation.
            SCVerify (bool, optional): Enables C-to-RTL SCVerify verification after HLS. Defaults to False.
            Synth (bool, optional): Enables the full HLS run, setting to False stops after HLS compilation.
            vhdl (bool, optional): Enables post-HLS VHDL netlist generation. Defaults to True.
            verilog (bool, optional): Enables post-HLS Verilog netlist generation. Defaults to True.
            RTLSynth (bool, optional): Enables downstream RTL synthesis. Default to False.
            RandomTBFrame (int, optional): In the absense of python dataset files for SCVerify, generate N frames of 
                random feature data. Defaults to 2.
            PowerEst (bool, optional): Enables post-HLS power estimation. Default to False.
            PowerOpt (bool, optional): Enables post-HLS power optimization. Default to False.
            BuildBUP (bool, optional): Enables a bottom-up HLS flow. Defaults to False.
            BUPWorkers (int, optional): When non-zero, specifies the number of parallel Catapult jobs to run. Defaults to 0.
            LaunchDA (bool, optional): Enables launching Catapult Design Analyzer during/after HLS. Defaults to False.

        Returns:
            dict: initial configuration.
        """
        config = {}
        config['Backend'] = 'Catapult'
        config['Technology'] = tech
        if tech == 'fpga':
            config['Part'] = part if part is not None else 'xcvu13p-flga2577-2-e'
        else:
            config['ASICLibs'] = asiclibs if asiclibs is not None else 'nangate-45nm'
        config['FIFO'] = fifo
        config['ASICFIFO'] = asicfifo
        config['ClockPeriod'] = clock_period
        config['FIFO'] = fifo
        config['IOType'] = io_type
        config['ProjectDir'] = project_dir
        config['HLSConfig'] = {}
        config['WriterConfig'] = {
            'Namespace': namespace,
            'WriteWeightsTxt': write_weights_txt,
            'WriteTar': write_tar,
        }
        config['ROMLocation'] = 'Local'
        config['CopyNNET'] = False
        # New experimental option
        config['CModelDefaultThreshold'] = 0.0
        config['BuildOptions'] = {
            'csim':           csim,
            'SCVerify':       SCVerify,
            'Synth':          Synth,
            'vhdl':           vhdl,
            'verilog':        verilog,
            'RTLSynth':       RTLSynth,
            'RandomTBFrames': RandomTBFrames,
            'PowerEst':       PowerEst,
            'PowerOpt':       PowerOpt,
            'BuildBUP':       BuildBUP,
            'BUPWorkers':     BUPWorkers,
            'LaunchDA':       LaunchDA
        }

        return config

    # Note: the following options are depricated and replaced
    #   cosim->SCVerify, validation->SCVerify, vsynth->rtlsyn
    # Note: the following options are not yet supported
    #   export, fifo_opt, bitfile
    def build(
        self,
        model,
        reset=False,
        csim=None,
        synth=None,
        cosim=None,
        validation=None,
        export=None,
        vsynth=None,
        fifo_opt=None,
        bitfile=None,
        sw_opt=None,
        power=None,
        da=None,
        bup=None,
        ran_frame=None,
        vhdl=None,
        verilog=None,
        Synth=None,
        SCVerify=None,
        RandomTBFrames=None,
        PowerEst=None,
        PowerOpt=None,
        LaunchDA=None,
        BuildBUP=None,
        BUPWorkers=None,
        RTLSynth=None,
    ):
        # Some argument checks/mappings
        if cosim is not None:
            print("HLS4ML build() option 'cosim' is being depricated. Use 'SCVerify'")
            SCVerify = cosim if cosim is True else SCVerify
        if validation is not None:
            print("HLS4ML build() option 'validation' is being depricated. Use 'SCVerify'")
            SCVerify = validation if validation is True else SCVerify
        if vsynth is not None:
            print("HLS4ML build() option 'vsynth' is being depricated. Use 'RTLSynth'")
            RTLSynth = vsynth if vsynth is True else RTLSynth
        if export is not None:
            print("HLS4ML build() option 'export' is not yet supported")
        if fifo_opt is not None:
            print("HLS4ML build() option 'fifo_opt' is not yet supported")
        if bitfile is not None:
            print("HLS4ML build() option 'bitfile' is not yet supported")

        if synth is not None:
            print("HLS4ML build() option 'synth' is being depricated. Use 'Synth'")
            Synth = synth if synth is True else Synth
        if ran_frame is not None:
            print("HLS4ML build() option 'ran_frame' is being depricated. Use 'RandomTBFrames'")
            RandomTBFrames = ran_frame if ran_frame is True else RandomTBFrames
        if sw_opt is not None:
            print("HLS4ML build() option 'sw_opt' is being depricated. Use 'PowerEst'")
            PowerEst = sw_opt if sw_opt is True else PowerEst
        if power is not None:
            print("HLS4ML build() option 'power' is being depricated. Use 'PowerOpt'")
            PowerOpt = power if power is True else PowerOpt
        if da is not None:
            print("HLS4ML build() option 'da' is being depricated. Use 'LaunchDA'")
            LaunchDA = da if da is True else LaunchDA
        if bup is not None:
            print("HLS4ML build() option 'bup' is being depricated. Use 'BuildBUP'")
            BuildBUP = bup if bup is True else BuildBUP

        # print(f'ran_frame value: {ran_frame}')  # Add this line for debugging
        catapult_exe = 'catapult'
        if 'linux' in sys.platform:
            cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            found = os.system(cmd)
            if found != 0:
                catapult_exe = os.getenv('MGC_HOME') + '/bin/catapult'
                cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            found = os.system(cmd)
            if found != 0:
                catapult_exe = os.getenv('CATAPULT_HOME') + '/bin/catapult'
                cmd = 'command -v ' + catapult_exe + ' > /dev/null'
            if found != 0:
                raise Exception('Catapult HLS installation not found. Make sure "catapult" is on PATH.')

        curr_dir = os.getcwd()
        # this execution moves into the hls4ml-generated "output_dir" and runs the build_prj.tcl script.
        os.chdir(model.config.get_output_dir())
        ccs_args = f'"reset={reset}'
        if csim is not None:
            ccs_args += f' csim={csim}'
        if vhdl is not None:
            ccs_args += f' vhdl={vhdl}'
        if verilog is not None:
            ccs_args += f' verilog={verilog}'
        if Synth is not None:
            ccs_args += f' Synth={Synth}'
        if SCVerify is not None:
            ccs_args += f' SCVerify={SCVerify}'
        if RandomTBFrames is not None:
            ccs_args += f' RandomTBFrames={RandomTBFrames}'
        if PowerEst is not None:
            ccs_args += f' PowerEst={PowerEst}'
        if PowerOpt is not None:
            ccs_args += f' PowerOpt={PowerOpt}'
        if LaunchDA is not None:
            ccs_args += f' LaunchDA={LaunchDA}'
        if BuildBUP is not None:
            ccs_args += f' BuildBUP={BuildBUP}'
        if BUPWorkers is not None:
            ccs_args += f' BUPWorkers={BUPWorkers}'
        if RTLSynth is not None:
            ccs_args += f' RTLSynth={RTLSynth}'
        ccs_args += '"'
        print(f'Catapult backend build() option overrides: {ccs_args}')
        ccs_invoke = catapult_exe + ' -product ultra -shell -f build_prj.tcl -eval \'set ::argv ' + ccs_args + '\''
        print(ccs_invoke)
        os.system(ccs_invoke)
        os.chdir(curr_dir)

        return parse_catapult_report(model.config.get_output_dir())

    def _validate_conv_strategy(self, layer):
        if layer.model.config.pipeline_style.lower() != 'dataflow':
            print(f'WARNING: Layer {layer.name} requires "dataflow" pipeline style. Switching to "dataflow" pipeline style.')
            layer.model.config.pipeline_style = 'dataflow'

    @layer_optimizer(Layer)
    def init_base_layer(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('reuse_factor', reuse_factor)

        target_cycles = layer.model.config.get_target_cycles(layer)
        layer.set_attr('target_cycles', target_cycles)

    @layer_optimizer(Dense)
    def init_dense(self, layer):
        index_t = IntegerPrecisionType(width=1, signed=False)
        compression = layer.model.config.get_compression(layer)
        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            if compression:
                layer.set_attr('strategy', 'compressed')
                index_t = layer.get_weights('weight').type.index_precision
            else:
                layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')
        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', index_t))

    # TODO consolidate these functions into a single `init_conv`
    @layer_optimizer(Conv1D)
    def init_conv1d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv1D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_target_reuse_factor(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_width = layer.get_output_variable().shape[0]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(1, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        self._validate_conv_strategy(layer)

    @layer_optimizer(SeparableConv1D)
    def init_sepconv1d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan_conv'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(Conv2D)
    def init_conv2d(self, layer):
        if len(layer.weights['weight'].data.shape) == 2:  # This can happen if we assign weights of Dense layer to 1x1 Conv2D
            layer.weights['weight'].data = np.expand_dims(layer.weights['weight'].data, axis=(0, 1))

        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            self.set_target_reuse_factor(layer)
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        out_height = layer.get_output_variable().shape[0]
        out_width = layer.get_output_variable().shape[1]
        chosen_pf = layer.model.config.get_layer_config_value(layer, 'ParallelizationFactor', 1)
        valid_pf = self.get_valid_conv_partition_splits(out_height, out_width)
        if chosen_pf not in valid_pf:
            closest_pf = self.get_closest_reuse_factor(valid_pf, chosen_pf)
            valid_pf_str = ','.join(map(str, valid_pf))
            print(
                f'WARNING: Invalid ParallelizationFactor={chosen_pf} in layer "{layer.name}".'
                f'Using ParallelizationFactor={closest_pf} instead. Valid ParallelizationFactor(s): {valid_pf_str}.'
            )
        else:
            closest_pf = chosen_pf
        layer.set_attr('n_partitions', out_height * out_width // closest_pf)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        self._validate_conv_strategy(layer)

    @layer_optimizer(SeparableConv2D)
    def init_sepconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_chan_conv'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    @layer_optimizer(DepthwiseConv2D)
    def init_depconv2d(self, layer):
        if layer.model.config.is_resource_strategy(layer):
            layer.set_attr('strategy', 'resource')
            n_in, n_out = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr(
            'n_partitions', 1
        )  # TODO Once we have SeparableConv implementation for io_parallel this should be set properly
        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

        # Set the output type of the depthwise phase
        dw_out_precision, _ = layer.model.config.get_precision(layer, 'dw_output')
        dw_out_name = layer.name + '_dw_out_t'
        if layer.model.config.get_config_value('IOType') == 'io_stream':
            dw_output_t = PackedType(dw_out_name, dw_out_precision, layer.get_attr('n_filt'), n_pack=1)
        else:
            dw_output_t = NamedType(dw_out_name, dw_out_precision)
        layer.set_attr('dw_output_t', dw_output_t)

    def _set_pooling_accum_t(self, layer, pool_size):
        extra_bits = ceil_log2(pool_size)
        accum_t = layer.get_attr('accum_t')
        accum_t.precision.width += extra_bits * 2
        if isinstance(accum_t.precision, FixedPrecisionType):
            accum_t.precision.integer += extra_bits

    @layer_optimizer(Pooling1D)
    def init_pooling1d(self, layer):
        pool_size = layer.get_attr('pool_width')
        self._set_pooling_accum_t(layer, pool_size)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(Pooling2D)
    def init_pooling2d(self, layer):
        pool_size = layer.get_attr('pool_height') * layer.get_attr('pool_width')
        self._set_pooling_accum_t(layer, pool_size)

        layer.set_attr('implementation', layer.model.config.get_conv_implementation(layer).lower())

    @layer_optimizer(GlobalPooling1D)
    def init_global_pooling1d(self, layer):
        pool_size = layer.get_attr('n_in')
        self._set_pooling_accum_t(layer, pool_size)

    @layer_optimizer(GlobalPooling2D)
    def init_global_pooling2d(self, layer):
        pool_size = layer.get_attr('in_height') * layer.get_attr('in_width')
        self._set_pooling_accum_t(layer, pool_size)

    @layer_optimizer(Softmax)
    def init_softmax(self, layer):
        if layer.model.config.get_config_value('IOType') == 'io_parallel':
            assert (
                len(layer.get_input_variable().shape) == 1
            ), 'Softmax with io_parallel strategy cannot be used on multidimensional tensors.'

    @layer_optimizer(Embedding)
    def init_embed(self, layer):
        if layer.attributes['n_in'] is None:
            raise Exception('Input length of Embedding layer must be specified.')

    @layer_optimizer(LSTM)
    def init_lstm(self, layer):
        # TODO Allow getting recurrent reuse factor from the config
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GRU)
    def init_gru(self, layer):
        reuse_factor = layer.model.config.get_reuse_factor(layer)
        layer.set_attr('recurrent_reuse_factor', reuse_factor)

        if layer.model.config.is_resource_strategy(layer):
            n_in, n_out, n_in_recr, n_out_recr = self.get_layer_mult_size(layer)
            self.set_closest_reuse_factor(layer, n_in, n_out)
            self.set_closest_reuse_factor(layer, n_in_recr, n_out_recr, attribute='recurrent_reuse_factor')
            layer.set_attr('strategy', 'resource')
        else:
            layer.set_attr('strategy', 'latency')

        layer.set_attr('index_t', NamedType(f'layer{layer.index}_index', IntegerPrecisionType(width=1, signed=False)))

    @layer_optimizer(GarNet)
    def init_garnet(self, layer):
        reuse_factor = layer.attributes['reuse_factor']

        var_converter = CatapultArrayVariableConverter(
            type_converter=HLSTypeConverter(precision_converter=ACTypeConverter())
        )

        # A bit controversial but we are going to set the partitioning of the input here
        in_layer = layer.model.graph[layer.inputs[0]]
        in_var = layer.get_input_variable(layer.inputs[0])
        partition_factor = in_var.shape[1] * (in_var.shape[0] // reuse_factor)
        in_pragma = ('partition', 'cyclic', partition_factor)
        new_in_var = var_converter.convert(in_var, pragma=in_pragma)
        in_layer.set_attr(layer.inputs[0], new_in_var)

        if layer.attributes['collapse']:
            out_pragma = 'partition'
        else:
            partition_factor = layer._output_features * (layer.attributes['n_vertices'] // reuse_factor)
            out_pragma = ('partition', 'cyclic', partition_factor)

        out_name, out_var = next(iter(layer.variables.items()))
        new_out_var = var_converter.convert(out_var, pragma=out_pragma)

        layer.set_attr(out_name, new_out_var)

    @layer_optimizer(GarNetStack)
    def init_garnet_stack(self, layer):
        self.init_garnet(layer)
