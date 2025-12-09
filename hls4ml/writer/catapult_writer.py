import glob
import os
import re
import stat
import tarfile
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np
import yaml

from hls4ml.backends import get_backend
from hls4ml.writer.writers import Writer
write_impl_type = None  # global definition

config_filename = 'hls4ml_config.yml'


class CatapultWriter(Writer):
    def print_array_to_cpp(self, var, odir, namespace=None, write_txt_file=True):
        """Write a weights array to C++ header files.

        Args:
            var (WeightVariable): Weight to write
            odir (str): Output directory
            namespace (str, optional): Writes a namespace for the weights to avoid clashes with global variables.
            write_txt_file (bool, optional): Write txt files in addition to .h files. Defaults to True.
        """

        h_file = open(f'{odir}/firmware/weights/{var.name}.h', 'w')
        if write_txt_file:
            txt_file = open(f'{odir}/firmware/weights/{var.name}.txt', 'w')

        # meta data
        h_file.write(f'//Numpy array shape {var.shape}\n')
        h_file.write(f'//Min {np.min(var.min):.12f}\n')
        h_file.write(f'//Max {np.max(var.max):.12f}\n')
        h_file.write(f'//Number of zeros {var.nzeros}\n')
        h_file.write('\n')

        h_file.write(f'#ifndef {var.name.upper()}_H_\n')
        h_file.write(f'#define {var.name.upper()}_H_\n')
        h_file.write('\n')

        if namespace is not None:
            h_file.write(f'namespace {namespace} {{\n\n')

        if write_txt_file:
            h_file.write('#ifndef __SYNTHESIS__\n')
            h_file.write('// global extern pointer only - actual array allocated in myproject_test.cpp\n')
            h_file.write('extern ' + var.definition_cpp() + ';\n')
            h_file.write('#else\n')

        h_file.write(var.definition_cpp() + ' = {')

        # fill c++ array.
        # not including internal brackets for multidimensional case
        sep = ''
        for x in var:
            h_file.write(sep + x)
            if write_txt_file:
                txt_file.write(sep + x)
            sep = ', '
        h_file.write('};\n\n')

        if write_txt_file:
            h_file.write('#endif\n')
            txt_file.close()

        if namespace is not None:
            h_file.write('}\n\n')

        h_file.write('\n#endif\n')
        h_file.close()

    def write_output_dir(self, model):
        """Write the base output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """
        if not os.path.isdir(f'{model.config.get_output_dir()}/firmware/weights'):
            os.makedirs(f'{model.config.get_output_dir()}/firmware/weights')

    @staticmethod
    def _make_array_pragma(variable, model):
        """
        Layers in hls_model.py can specify output array partitioning through the `pragma` attribute.
        If `pragma` is a string: options are 'partition', 'reshape', or 'stream'.
        If `pragma` is a tuple: (mode, type, factor) where mode is 'partition' or 'reshape', type is
        'complete', 'cyclic', or 'block', and factor is an integer only used when the type is not 'complete'.
        """

        # Walk model looking for any layer reconvergence (such cases would preclude setting FIFO_DEPTH=1)
        # (Not the most efficient algorithm (called for every variable and cut-n-pasted in _make_array_fifo_pragma below)
        no_reconvergence = True
        fifo_depth = model.config.get_config_value("FIFO_DEPTH", default=1) 
        for layer in model.get_layers():
            if layer.attributes.layer.class_name == 'Concatenate':
                no_reconvergence = False

        config = variable.pragma
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode in ['partition', 'reshape']:
            if typ == 'complete':
                template = '// {mode} variable={name} {type} dim={dim}'
            else:
                template = '// {mode} variable={name} {type} factor={factor} dim={dim}'

            return template.format(mode=mode.upper(), name=variable.name, type=typ, factor=factor, dim=0)

        elif mode == 'stream':
            fifo = None
            tech = model.config.get_config_value("Technology")
            if tech is not None:
                if tech == 'asic':
                    fifo = model.config.get_config_value("ASICFIFO")
                else:
                    fifo = model.config.get_config_value("FIFO")
            if fifo is None:
                fifo = model.config.get_config_value("FIFO")
            if fifo is not None:
                retstr = f'#pragma hls_resource {variable.name}:cns variables="{variable.name}"'
                if no_reconvergence:
                    retstr += f' map_to_module="{fifo}" fifo_depth="1"'
                else:
                    retstr += f' map_to_module="{fifo}" fifo_depth="{depth}"'
                return retstr
            else:
                return ''
        else:
            return ''

    @staticmethod
    def _make_array_fifo_pragma(variable, model):

        # Walk model looking for any layer reconvergence (such cases would preclude setting FIFO_DEPTH=1)
        # (Not the most efficient algorithm (called for every variable and cut-n-pasted in _make_array_fifo_pragma below)
        no_reconvergence = True
        for layer in model.get_layers():
            if layer.attributes.layer.class_name == 'Concatenate':
                no_reconvergence = False

        config = variable.pragma
        factor = ''
        if type(config) is tuple:
            mode = config[0]
            if mode in ['partition', 'reshape']:
                typ = config[1]
                if typ != 'complete':
                    factor = config[2]
            elif mode == 'stream':
                depth = config[1]
        else:
            mode = config
            typ = 'complete'
            factor = 0

        if mode == 'stream':
            fifo = None
            tech = model.config.get_config_value("Technology")
            if tech is not None:
                if tech == 'asic':
                    fifo = model.config.get_config_value("ASICFIFO")
                else:
                    fifo = model.config.get_config_value("FIFO")
            if fifo is None:
                fifo = model.config.get_config_value("FIFO")

            if fifo is not None:
                if no_reconvergence:
                    return f'// #pragma hls_fifo_depth 1 {factor}'
                else:
                    return f'// #pragma hls_fifo_depth {depth} {factor}'
            else:
                return ''
        else:
            return ''

    def write_project_cpp(self, model, memory_type, port_type):
        """Write the main architecture source file (myproject.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        ROMLocation = model.config.get_config_value('ROMLocation')
        filedir = os.path.dirname(os.path.abspath(__file__))

        f = open(os.path.join(filedir, '../templates/catapult/firmware/myproject.cpp'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '
        
        #--------------------------------------------
        # - Support for Programmable shared weights
        total_size = 0
        merged_array_str = ""
        ac_shared_declarations = ""
        load_scratchpad_fn = ""
        load_call = 'load_scratchpad(reload, weights_biases'
        # Define the weights_biases array, assuming we have a total size for all weights
        weight_arrays = []  # To store weight variable details
        sync_variables = []  # To store dynamic sync variable names
        resource_pragmas = []  # To store resource pragmas for weights
        if memory_type == 'RAM':
            print('RAM not supported')
        #--------------------------------------------

        for line in f.readlines():
            # Add headers to weights and biases
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                io_type = model.config.get_config_value('IOType')
                if io_type == 'io_parallel' :
                    inputs_str = ', '.join([i.definition_cpp(as_reference=True) + ', ac_sync &' + i.name + '_sync' for i in model_inputs])
                    outputs_str = ', '.join([o.definition_cpp(as_reference=True) + ', ac_sync &' + o.name + '_sync'for o in model_outputs])
                else :
                    inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                    outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str
                if len(model_brams) > 0:
                    # Check if the memory_type is RAM
                    if memory_type == 'RAM':
                        print('RAM not supported')
                    else:
                        newline += ',\n' + brams_str
                newline += '\n'

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            elif (ROMLocation=='Local') and ('// hls-fpga-machine-learning insert weights' in line):
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != 'bram':
                            newline += f'#include "weights/{w.name}.h"\n'

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.weight_class == 'CompressedWeightVariable':
                            newline += indent + '    nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.nonzeros, w.name, w.name
                            )
                        elif w.weight_class == 'ExponentWeightVariable':
                            newline += indent + '    nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                w.type.name, w.data_length, w.name, w.name
                            )
                        else:
                            if memory_type == 'RAM':
                                print('RAM not supported')
                            else:
                                # Keep the original line for other memory types
                                newline += indent + '    nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

            # Add Interface Synthesis resource pragmas
            elif '// hls-fpga-machine-learning insert IFSynPragmas' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                io_type = model.config.get_config_value('IOType')
                tech = model.config.get_config_value('Technology')
                if tech == 'asic':
                    mem_lib = model.config.get_config_value('ASICRAM')
                else:
                    mem_lib = model.config.get_config_value('RAM')
                
                if io_type == 'io_serial' or io_type == 'io_stream':
                    # Eventually this will be amba.ccs_axi4stream_in and amba.ccs_axi4stream_out
                    if memory_type == 'RAM':
                        print('RAM not supported')
                    for dut_input in all_inputs:
                        newline += f'#pragma hls_resource {dut_input}:rsc variables="{dut_input}"'
                        newline += ' map_to_module="ccs_ioport.ccs_in_wait"\n'
                    for dut_output in all_outputs:
                        newline += f'#pragma hls_resource {dut_output}:rsc variables="{dut_output}"'
                        newline += ' map_to_module="ccs_ioport.ccs_out_wait"\n'
                    
            # Add input/output type
            elif '// hls-fpga-machine-learning insert IO' in line:
                newline = line
                all_inputs = [i.name for i in model_inputs]
                all_outputs = [o.name for o in model_outputs]
                all_brams = [b.name for b in model_brams]
                io_type = model.config.get_config_value('IOType')

                pipeline_style = model.config.pipeline_style
                pipeline_ii = model.config.pipeline_ii
                pipeline_pragma = indent + f'#pragma HLS {pipeline_style.upper()}'
                if pipeline_style == 'pipeline' and pipeline_ii is not None:
                    pipeline_pragma += f' II={pipeline_ii}\n'
                else:
                    pipeline_pragma += '\n'

                if io_type == 'io_parallel':
                    for i in model_inputs:
                        newline += indent + self._make_array_pragma(i, model) + '\n'
                    for o in model_outputs:
                        newline += indent + self._make_array_pragma(o, model) + '\n'

            elif '// hls-fpga-machine-learning insert layers' in line:
                io_type = model.config.get_config_value('IOType')
                newline = line + '\n'
                declared_vars = set()  # Track already declared variables

                for layer in model.get_layers():
                    vars = layer.get_variables()
                    for var in vars:
                        if var not in model_inputs and var not in model_outputs:
                            def_cpp = var.definition_cpp()
                            if def_cpp is not None and def_cpp not in declared_vars:  # Check for duplicates
                                declared_vars.add(def_cpp)  # Mark as declared
                                if var.pragma:
                                    newline += '    ' + self._make_array_fifo_pragma(var, model) + '\n'
                                if io_type == 'io_serial' or io_type == 'io_stream':
                                    newline += '    static ' + def_cpp + '; \n'
                                else:
                                    if type(var).__name__ == 'CatapultArrayVariable':
                                        #shared_def_cpp = var.ac_shared_definition_cpp()
                                        tmpF = layer.get_attr('function_cpp', None)
                                        if tmpF :
                                            shared_def_cpp = f'ac_shared<{var.type.name}[{var.size_cpp()}] > {var.name} /* {var.pragma} */'
                                            init_str = f'ac::init_array<AC_VAL_DC>({var.name}, {var.size_cpp()})'
                                            
                                            newline += '    ' + 'static ' + shared_def_cpp + '; \n'
                                            newline += '    ' + 'static bool ' + var.name + '_init = ' + init_str + '; \n'
                                            newline += '    ' + 'static ac_sync ' + var.name + '_sync;\n'
                                        else:
                                            inVar = var.input_var;
                                            if inVar in model_inputs :
                                                newline += '    ' + var.type.name  + ' *' + var.name + ' = ' + inVar.name +'; \n'                                                
                                            else :
                                                shared_def_cpp = f'ac_shared<{var.type.name}[{var.size_cpp()}] > &{var.name} '
                                                newline += '    ' + 'static ' + shared_def_cpp + ' = ' + inVar.name +'; \n'

                                            newline += '    ' + 'static ac_sync &' + var.name + '_sync = ' + inVar.name + '_sync;\n'
                                    else:
                                        newline += '    ' + def_cpp + '; \n'
                                if var.pragma:
                                    newline += '    ' + self._make_array_pragma(var, model) + '\n'
                    func = layer.get_attr('function_cpp', None)
                    if memory_type == 'RAM':
                        print('RAM not supported')
                        
                    if func and io_type == 'io_parallel':
                        ## Fixup the func string with inserted sync signals.
                        # Collect layer I/O
                        l_inputs = []
                        if 'inputs' in layer.attributes:
                            for name in layer.attributes['inputs']:
                                var = model.graph[name].get_output_variable()
                                l_inputs += [v.name for v in var] if isinstance(var, list) else [var.name]
                        else:
                            var = layer.get_input_variable()
                            l_inputs = [v.name for v in var] if isinstance(var, list) else ([var.name] if var else [])
 
                        var = layer.get_output_variable()
                        l_outputs = [v.name for v in var] if isinstance(var, list) else ([var.name] if var else [])
                        for inp in l_inputs:
                            func = func.replace(f"{inp}", f"{inp}, {inp}_sync")
                        for outp in l_outputs:
                            func = func.replace(f"{outp}", f"{outp}, {outp}_sync")
                    if func:
                        if not isinstance(func, (list, set)):
                            func = [func]
                        if len(func) == 1:
                            newline += '    ' + func[0] + ' // ' + layer.name + '\n'
                        else:
                            newline += '    // ' + layer.name + '\n'
                            for line in func:
                                newline += '    ' + line + '\n'
                        if model.config.trace_output and layer.get_attr('trace', False):
                            newline += '#ifndef __SYNTHESIS__\n'
                            for var in vars:
                                newline += '    nnet::save_layer_output<{}>({}, "{}", {});\n'.format(
                                    var.type.name, var.name, layer.name, var.size_cpp()
                                )
                            newline += '#endif\n'
                        newline += '\n'

            # Just copy line
            else:
                newline = line

            fout.write(newline)
        f.close()
        fout.close()

    def write_project_header(self, model, memory_type, port_type):
        """Write the main architecture header file (myproject.h)

        Args:
            model (ModelGraph): the hls4ml model.
            memory_type (str): the type of memory (e.g., 'RAM', 'BRAM', etc.).
            port_type (str): the type of port (e.g., 'ac_fixed', 'ac_channel', etc.).
        """

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/myproject.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/{model.config.get_project_name()}.h', 'w')

        # Gather model inputs, outputs, and BRAM-stored variables
        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        # Calculate total size of weights and biases by multiplying dimensions
        model_weights = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']
        total_weights_and_biases = sum([np.prod(var.shape) for var in model_weights])

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert header' in line:
                io_type = model.config.get_config_value('IOType')
                if io_type == 'io_parallel' :
                    inputs_str = ', '.join([i.definition_cpp(as_reference=True) + ', ac_sync &' + i.name + '_sync' for i in model_inputs])
                    outputs_str = ', '.join([o.definition_cpp(as_reference=True) + ', ac_sync &' + o.name + '_sync'for o in model_outputs])
                else :
                    inputs_str = ', '.join([i.definition_cpp(as_reference=True) for i in model_inputs])
                    outputs_str = ', '.join([o.definition_cpp(as_reference=True) for o in model_outputs])
                brams_str = ', \n'.join([indent + b.definition_cpp(as_reference=False) for b in model_brams])

                # Check if memory_type is RAM to merge weights and biases
                if memory_type == 'RAM':
                    print('RAM not supported')
                else:
                    # Otherwise, keep individual weight and bias arrays
                    newline = ''
                    newline += indent + inputs_str + ',\n'
                    newline += indent + outputs_str
                    if len(model_brams) > 0:
                        newline += ',\n' + brams_str
                    newline += '\n'

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''
                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''
                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            else:
                newline = line

            fout.write(newline)

        f.close()
        fout.close()

    def write_defines(self, model):
        """Write the C++ type definitions file (defines.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/defines.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/defines.h', 'w')

        bus_words = model.config.get_config_value('AC_BUS_WORDS', default=None)

        for line in f.readlines():
            # Insert numbers
            if '// hls-fpga-machine-learning insert numbers' in line:
                newline = line

                # if implementation == 'ac_window':
                if bus_words is None:
                    bus_words = 1
                newline += f'enum {{\n  AC_BUS_WORDS = {bus_words},\n}};\n\n'

            elif '// hls-fpga-machine-learning insert layer-precision' in line:
                newline = line
                all_precision = OrderedDict()
                for layer in model.get_layers():
                    layer_precision = layer.get_layer_precision()
                    for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types doesn't override existing types
                        # This can happen in case of InplaceVariable types
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var
                for used_type in all_precision.values():
                    newline += used_type.definition_cpp()

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_parameters(self, model):
        """Write the C++ layer config file (parameters.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        ROMLocation = model.config.get_config_value('ROMLocation')
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/firmware/parameters.h'))
        fout = open(f'{model.config.get_output_dir()}/firmware/parameters.h', 'w')

        for line in f.readlines():
            if '// hls-fpga-machine-learning insert includes' in line:
                newline = line
                for include in sorted(set(sum((layer.get_attr('include_header', []) for layer in model.get_layers()), []))):
                    newline += '#include "%s"\n' % include

            elif (ROMLocation!='Local') and ('// hls-fpga-machine-learning insert weights' in line):
                newline = line
                for layer in model.get_layers():
                    for w in layer.get_weights():
                        if w.storage.lower() != 'bram':
                            newline += f'#include "weights/{w.name}.h"\n'

            elif '// hls-fpga-machine-learning insert layer-config' in line:
                newline = line
                for layer in model.get_layers():
                    config = layer.get_attr('config_cpp', None)
                    if config:
                        newline += '// ' + layer.name + '\n'
                        newline += config + '\n'

            elif '// hls-fpga-machine-learning insert namespace-start' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += f'namespace {namespace} {{\n'

            elif '// hls-fpga-machine-learning insert namespace-end' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += '}\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_weights(self, model):
        """Write the weights into header files

        Args:
            model (ModelGraph): the hls4ml model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for layer in model.get_layers():
            for weights in layer.get_weights():
                self.print_array_to_cpp(
                    weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                )

    def write_multigraph_weights(self, model):
        """Write the weights into header files

        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        namespace = model.config.get_writer_config().get('Namespace', None)
        write_txt = model.config.get_writer_config().get('WriteWeightsTxt', True)
        for g in model.graphs:
            for layer in g.get_layers():
                for weights in layer.get_weights():
                    self.print_array_to_cpp(
                        weights, model.config.get_output_dir(), namespace=namespace, write_txt_file=write_txt
                    )

    def __make_dat_file(self, original_path, project_path):
        """
        Convert other input/output data types into a dat file, which is
        a text file with the falttened matrix printed out. Note that ' ' is
        assumed to be the delimiter.
        """

        # Take in data from current supported data files
        if original_path[-3:] == 'npy':
            data = np.load(original_path)
        else:
            raise Exception('Unsupported input/output data files.')

        # Faltten data, just keep first dimension
        data = data.reshape(data.shape[0], -1)

        def print_data(f):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write(str(data[i][j]) + ' ')
                f.write('\n')

        # Print out in dat file
        with open(project_path, 'w') as f:
            print_data(f)

    def write_test_bench(self, model, memory_type, port_type):
        """Write the testbench files (myproject_test.cpp and input/output .dat files)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        io_type = model.config.get_config_value('IOType')
        filedir = os.path.dirname(os.path.abspath(__file__))
        impl = write_impl_type

        if not os.path.exists(f'{model.config.get_output_dir()}/tb_data/'):
            os.mkdir(f'{model.config.get_output_dir()}/tb_data/')

        input_data = model.config.get_config_value('InputData')
        output_predictions = model.config.get_config_value('OutputPredictions')

        if input_data and os.path.exists(input_data):
            if input_data[-3:] == 'dat':
                copyfile(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')
            else:
                self.__make_dat_file(input_data, f'{model.config.get_output_dir()}/tb_data/tb_input_features.dat')

        if output_predictions and os.path.exists(output_predictions):
            if output_predictions[-3:] == 'dat':
                copyfile(output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat')
            else:
                self.__make_dat_file(
                    output_predictions, f'{model.config.get_output_dir()}/tb_data/tb_output_predictions.dat'
                )

        f = open(os.path.join(filedir, '../templates/catapult/myproject_test.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_test.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        for line in f.readlines():
            indent = ' ' * (len(line) - len(line.lstrip(' ')))

            # Insert numbers
            if 'myproject' in line:
                newline = line.replace('myproject', model.config.get_project_name())

            elif '// hls-fpga-machine-learning insert catapult_scverify' in line:
                newline += f'#include \"nnet_utils/nnet_scverify.h\"\n'

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert declare weights' in line:
                newline = line
                total_size = 0  # Initialize total weight counter

                if memory_type == 'RAM':  # Check if memory type is RAM
                    print('RAM not supported')
                else:
                    # Execute the old code (from comments)
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            newline += w.definition_cpp() + ";\n"

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                offset = 0  # Initialize offset for the weights_biases array

                if memory_type == 'RAM':  # Check if the memory type is 'RAM'
                    print('RAM not supported')
                else:
                    # Original code logic for non-RAM memory types
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            if w.weight_class == 'CompressedWeightVariable':
                                newline += indent + 'nnet::load_compressed_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.nonzeros, w.name, w.name
                                )
                            elif w.weight_class == 'ExponentWeightVariable':
                                newline += indent + 'nnet::load_exponent_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )
                            else:
                                newline += indent + 'nnet::load_weights_from_txt<{}, {}>({}, "{}.txt");\n'.format(
                                    w.type.name, w.data_length, w.name, w.name
                                )

            elif '// hls-fpga-machine-learning insert data' in line:
                newline = line
                offset = 0
                for inp in model_inputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{inp.type.name}[{inp.size_cpp()}] > {inp.name} /* {inp.pragma} */'
                        init_str = f'ac::init_array<AC_VAL_0>({inp.name}, {inp.size_cpp()})'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'static bool ' + inp.name + '_init = ' + init_str + '; \n'
                        newline += indent + 'ac_sync ' + inp.name + '_sync; \n'
                        newline += indent + inp.name + '_sync.sync_out(); \n'
                    else :
                        newline += indent + inp.definition_cpp() + ';\n'
                    # newline += indent + 'nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                    #     inp.type.name, offset, inp.size_cpp(), inp.name)
                    if impl == 'ac_window':
                        newline += indent + 'nnet::copy_data<float, {}, {}, {}, AC_BUS_WORDS>(in, {});\n'.format(
                            inp.type.name, offset, inp.size_cpp(), inp.name)
                    else:
                        newline += indent + 'nnet::copy_data<float, {}, {}, {}>(in, {});\n'.format(
                            inp.type.name, offset, inp.size_cpp(), inp.name)    
                    offset += inp.size()
                for out in model_outputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{out.type.name}[{out.size_cpp()}] > {out.name} /* {out.pragma} */'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'ac_sync ' + out.name + '_sync;\n'
                    else :
                        newline += indent + out.definition_cpp() + ';\n'
                        
            elif '// hls-fpga-machine-learning insert random' in line:
                newline = line
                for inp in model_inputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{inp.type.name}[{inp.size_cpp()}] > {inp.name} /* {inp.pragma} */'
                        init_str = f'ac::init_array<AC_VAL_DC>({inp.name}, {inp.size_cpp()})'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'static bool ' + inp.name + '_init = ' + init_str + '; \n'
                        newline += indent + 'ac_sync ' + inp.name + '_sync; \n'
                        newline += indent + inp.name + '_sync.sync_out(); \n'
                    else :
                        newline += indent + inp.definition_cpp() + ';\n'
                        # newline += indent + f'nnet::fill_random<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                        if impl == 'ac_window':
                            newline += indent + f'nnet::fill_random<{inp.type.name}, {inp.size_cpp()}, AC_BUS_WORDS>({inp.name});\n'
                        else:
                            newline += indent + f'nnet::fill_random<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n' 
                for out in model_outputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{out.type.name}[{out.size_cpp()}] > {out.name} /* {out.pragma} */'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'ac_sync ' + out.name + '_sync;\n'
                    else :
                        newline += indent + out.definition_cpp() + ';\n'

            elif '// hls-fpga-machine-learning insert zero' in line:
                newline = line
                for inp in model_inputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{inp.type.name}[{inp.size_cpp()}] > {inp.name} /* {inp.pragma} */'
                        init_str = f'ac::init_array<AC_VAL_0>({inp.name}, {inp.size_cpp()})'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'static bool ' + inp.name + '_init = ' + init_str + '; \n'
                        newline += indent + 'ac_sync ' + inp.name + '_sync; \n'
                        newline += indent + inp.name + '_sync.sync_out(); \n'
                    else :
                        newline += indent + inp.definition_cpp() + ';\n'
                        # newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n'
                        if impl == 'ac_window':
                            newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}, AC_BUS_WORDS>({inp.name});\n'
                        else:
                            newline += indent + f'nnet::fill_zero<{inp.type.name}, {inp.size_cpp()}>({inp.name});\n' 
                for out in model_outputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{out.type.name}[{out.size_cpp()}] > {out.name} /* {out.pragma} */'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'ac_sync ' + out.name + '_sync;\n'
                    else :
                        newline += indent + out.definition_cpp() + ';\n'

            elif '// hls-fpga-machine-learning insert top-level-function' in line:
                newline = line
                if io_type == 'io_parallel' :
                    input_vars = ', '.join([i.name + ', ' + i.name + '_sync' for i in model_inputs])
                    output_vars = ', '.join([o.name + ', ' + o.name + '_sync'for o in model_outputs])
                else :
                    input_vars = ','.join([i.name for i in model_inputs])
                    output_vars = ','.join([o.name for o in model_outputs])                    
                bram_vars = ','.join([b.name for b in model_brams])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                if memory_type=='RAM':
                    print('RAM not supported')
                else:
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))
                    
                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'

                newline += top_level
                if io_type == 'io_parallel' :
                    for out in model_outputs:
                        newline += indent + out.name + '_sync.sync_in();\n'

            elif '// hls-fpga-machine-learning insert output-compare' in line:
                newline = line
                offset_out = 0
                for out in model_outputs:
                    # newline += indent + 'total_err_cnt += nnet::compare_data<float, {}, {}, {}>(pr, {}, threshold);\n'.format(
                    #    out.type.name, offset_out, out.size_cpp(), out.name
                    # )
                    if impl == 'ac_window':
                        newline += indent + 'total_err_cnt += nnet::compare_data<float, {}, {}, {}, AC_BUS_WORDS>(pr, {}, threshold);\n'.format(
                        out.type.name, offset_out, out.size_cpp(), out.name
                        )
                    else:
                        newline += indent + 'total_err_cnt += nnet::compare_data<float, {}, {}, {}>(pr, {}, threshold);\n'.format(
                        out.type.name, offset_out, out.size_cpp(), out.name
                        )
                    offset_out += out.size()

            elif '// hls-fpga-machine-learning insert predictions' in line:
                newline = line
                for out in model_outputs:
                    newline += indent + f'for(int i = 0; i < {out.size_cpp()}; i++) {{\n'
                    newline += indent + '  std::cout << pr[i] << " ";\n'
                    newline += indent + '}\n'
                    newline += indent + 'std::cout << std::endl;\n'

            elif '// hls-fpga-machine-learning insert tb-output' in line:
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                if tb_stream != 'stdout':
                    for out in model_outputs:
                        # newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                        #     out.type.name, out.size_cpp(), out.name
                        # )  # TODO enable this
                        if impl == 'ac_window':
                            newline += indent + 'nnet::print_result<{}, {}, AC_BUS_WORDS>({}, fout);\n'.format(
                                out.type.name, out.size_cpp(), out.name)
                        else:
                            newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                                out.type.name, out.size_cpp(), out.name)

            elif (
                '// hls-fpga-machine-learning insert output' in line
                or '// hls-fpga-machine-learning insert quantized' in line
            ):
                newline = line
                tb_stream = model.config.get_writer_config().get('TBOutputStream', 'both')
                keep_output = str(tb_stream != 'stdout').lower()  # We keep output if we need to write it to file too.
                if tb_stream != 'file':
                    for out in model_outputs:
                        # newline += indent + 'nnet::print_result<{}, {}>({}, std::cout, true);\n'.format(
                        #     out.type.name, out.size_cpp(), out.name
                        # )
                        if impl == 'ac_window':
                            newline += indent + 'nnet::print_result<{}, {}, AC_BUS_WORDS>({}, fout);\n'.format(
                                out.type.name, out.size_cpp(), out.name)
                        else:
                            newline += indent + 'nnet::print_result<{}, {}>({}, fout);\n'.format(
                                out.type.name, out.size_cpp(), out.name)

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def write_bridge(self, model, memory_type, port_type):
        """Write the Python-C++ bridge (myproject_bridge.cpp)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        io_type = model.config.get_config_value('IOType')
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        model_brams = [var for var in model.get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))

            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            # Need to test this
            elif '// hls-fpga-machine-learning insert weights dir' in line:
                weights_dir = (Path(fout.name).parent / 'firmware/weights').resolve()
                newline = f'static std::string s_weights_dir = "{weights_dir}";\n'

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                if memory_type != 'RAM':
                    print('RAM not supported')

            elif '// hls-fpga-machine-learning insert declare weights' in line:
                newline = line

                if memory_type == 'RAM':  # Check if memory type is RAM
                    print('RAM not supported')
                else:
                    # Execute the old code (from comments)
                    for layer in model.get_layers():
                        for w in layer.get_weights():
                            newline += w.definition_cpp() + ";\n"

            elif '// hls-fpga-machine-learning insert load weights' in line:
                newline = line
                offset = 0  # Initialize offset for the weights_biases array
                if memory_type == 'RAM':  # Check if the memory type is 'RAM'
                    print('RAM not supported')

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])
                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{i.type.name}[{i.size_cpp()}] > {i.name}_ap /* {i.pragma} */'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'ac_sync ' + i.name + '_sync; \n'
                        newline += indent + i.name + '_sync.sync_out(); \n'
                    else :
                        newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    # newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                    #     dtype, i.type.name, i.size_cpp(), i.name, i.name)
                    impl = write_impl_type
                    if impl == 'ac_window':
                        newline += indent + 'nnet::convert_data<{}, {}, {}, AC_BUS_WORDS>({}, {}_ap);\n'.format(
                            dtype, i.type.name, i.size_cpp(), i.name, i.name)
                    else:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                            dtype, i.type.name, i.size_cpp(), i.name, i.name)

                newline += '\n'

                for o in model_outputs:
                    if io_type == 'io_parallel' :
                        shared_def_cpp = f'ac_shared<{o.type.name}[{o.size_cpp()}] > {o.name}_ap /* {o.pragma} */'
                        newline += indent + shared_def_cpp + ';\n'
                        newline += indent + 'ac_sync ' + o.name + '_sync;\n'
                    else :
                        newline += indent + '{var};\n'.format(var=o.definition_cpp(name_suffix='_ap'))

                newline += '\n'
                if memory_type=='RAM':
                    print('RAM not supported')

                if io_type == 'io_parallel' :
                    input_vars = ', '.join([i.name + '_ap, ' + i.name + '_sync' for i in model_inputs])
                    output_vars = ', '.join([o.name + '_ap, ' + o.name + '_sync'for o in model_outputs])
                else :
                    input_vars = ','.join([i.name + '_ap' for i in model_inputs])
                    output_vars = ','.join([o.name + '_ap' for o in model_outputs])
                bram_vars = ','.join([b.name for b in model_brams])

                # Concatenate the input, output, and bram variables. Filter out empty/null values
                if memory_type=='RAM':
                    print('RAM not supported')
                else: 
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))

                top_level = indent + f'{model.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    # newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                    #     o.type.name, dtype, o.size_cpp(), o.name, o.name
                    # )
                    impl = write_impl_type

                    if impl == 'ac_window':
                        newline += indent + 'nnet::convert_data<{}, {}, {}, AC_BUS_WORDS>({}_ap, {});\n'.format(
                            o.type.name, dtype, o.size_cpp(), o.name, o.name)
                    else:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                            o.type.name, dtype, o.size_cpp(), o.name, o.name)

            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_bridge_multigraph(self, model):
        """Write the Python-C++ bridge (myproject_stitched_bridge.cpp)
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """

        # TODO - make this align with write_bridge customizations
        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir, '../templates/catapult/myproject_bridge.cpp'))
        fout = open(f'{model.config.get_output_dir()}/{model.config.get_project_name()}_bridge.cpp', 'w')
        model_inputs = model.graphs[0].get_input_variables()
        model_outputs = model.graphs[-1].get_output_variables()
        model_brams = [var for var in model.graphs[0].get_weight_variables() if var.storage.lower() == 'bram']

        indent = '    '

        for line in f.readlines():
            newline = ''
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT', format(model.config.get_project_name().upper()))
            elif 'firmware/myproject' in line:
                for graph_idx, g in enumerate(model.graphs):
                    newline += '#undef DEFINES_H_\n'
                    if len(g.outputs) == 1:
                        newline += '#define result_t ' + 'result_graph' + str(graph_idx + 1) + '_t\n'
                    newline += line.replace('myproject', format(model.graphs[graph_idx].config.get_project_name()))
                    if len(g.outputs) == 1:
                        newline += (
                            'typedef result_graph' + str(graph_idx + 1) + '_t graph' + str(graph_idx + 1) + '_result_t;\n'
                        )
                        newline += '#undef result_t\n\n' if graph_idx < len(model.graphs) - 1 else '\n'
                newline += '\n'
            elif 'myproject' in line:
                newline = line.replace('myproject', format(model.config.get_project_name()))

            elif '// hls-fpga-machine-learning insert bram' in line:
                newline = line
                for bram in model_brams:
                    newline += f'#include \"firmware/weights/{bram.name}.h\"\n'

            elif '// hls-fpga-machine-learning insert header' in line:
                dtype = line.split('#', 1)[1].strip()
                inputs_str = ', '.join([f'{dtype} {i.name}[{i.size_cpp()}]' for i in model_inputs])
                outputs_str = ', '.join([f'{dtype} {o.name}[{o.size_cpp()}]' for o in model_outputs])

                newline = ''
                newline += indent + inputs_str + ',\n'
                newline += indent + outputs_str + '\n'

            elif '// hls-fpga-machine-learning insert wrapper' in line:
                dtype = line.split('#', 1)[1].strip()
                newline = ''
                for i in model_inputs:
                    newline += indent + '{var};\n'.format(var=i.definition_cpp(name_suffix='_ap'))
                    newline += indent + 'nnet::convert_data<{}, {}, {}>({}, {}_ap);\n'.format(
                        dtype, i.type.name, i.size_cpp(), i.name, i.name
                    )
                newline += '\n'

                for idx, g in enumerate(model.graphs):
                    for o in g.get_output_variables():
                        definition = o.definition_cpp(name_suffix='_ap')
                        if len(g.outputs) == 1:
                            parts = definition.split(' ', 1)
                            datatype = 'graph' + str(idx + 1) + '_result_t'
                            if parts[0].startswith('hls::stream'):
                                modified_definition = 'hls::stream<' + datatype + '> ' + parts[1]
                            else:
                                modified_definition = datatype + ' ' + parts[1]
                            newline += indent + f'{modified_definition};\n'
                        else:
                            newline += indent + f'{definition};\n'

                newline += '\n'

                top_level = ''
                output_vars = ''
                for idx, g in enumerate(model.graphs):
                    if idx == 0:
                        input_vars = ','.join([i.name + '_ap' for i in g.get_input_variables()])
                    else:
                        input_vars = output_vars
                    bram_vars = ','.join(
                        [b.name for b in [var for var in g.get_weight_variables() if var.storage.lower() == 'bram']]
                    )
                    output_vars = ','.join([o.name + '_ap' for o in g.get_output_variables()])
                    # Concatenate the input, output, and bram variables. Filter out empty/null values
                    all_vars = ','.join(filter(None, [input_vars, output_vars, bram_vars]))
                    top_level += indent + f'{g.config.get_project_name()}({all_vars});\n'
                newline += top_level

                newline += '\n'

                for o in model_outputs:
                    if len(model.graphs[-1].outputs) == 1:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                            datatype, dtype, o.size_cpp(), o.name, o.name
                        )
                    else:
                        newline += indent + 'nnet::convert_data<{}, {}, {}>({}_ap, {});\n'.format(
                            o.type.name, dtype, o.size_cpp(), o.name, o.name
                        )

            elif '// hls-fpga-machine-learning insert trace_outputs' in line:
                newline = ''
                for layer in model.get_layers():
                    func = layer.get_attr('function_cpp', None)
                    if func and model.config.trace_output and layer.get_attr('trace', False):
                        vars = layer.get_variables()
                        for var in vars:
                            newline += (
                                indent
                                + 'nnet::trace_outputs->insert(std::pair<std::string, void *>('
                                + f'"{layer.name}", (void *) malloc({var.size_cpp()} * element_size)));\n'
                            )

            elif '// hls-fpga-machine-learning insert namespace' in line:
                newline = ''

                namespace = model.config.get_writer_config().get('Namespace', None)
                if namespace is not None:
                    newline += indent + f'using namespace {namespace};\n'

            elif '// hls-fpga-machine-learning insert tb_input_writer' in line:
                funcs = [
                    ('float', 'dump_tb_inputs_float'),
                    ('double', 'dump_tb_inputs_double'),
                ]
                newline = ''
                for dtype, funcname in funcs:
                    newline += f'void {funcname}(\n'
                    newline += '    const char* output_path'
                    for inp in model_inputs:
                        newline += f',\n    {dtype} {inp.name}[{inp.size_cpp()}]'
                    newline += '\n) {\n\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        ap = inp.name + '_ap'
                        if decl.startswith("hls::stream"):
                            newline += f'    {decl};\n'
                        else:
                            newline += f'    {inp.type.name} {ap}[{inp.size_cpp()}];\n'
                        newline += (
                            f'    nnet::convert_data<{dtype}, {inp.type.name}, {inp.size_cpp()}>' f'({inp.name}, {ap});\n'
                        )
                    newline += "\n"
                    newline += f'    std::ofstream fout(std::string(output_path) + "/{inp.name}_input_data.txt");\n'

                    for inp in model_inputs:
                        decl = inp.definition_cpp(name_suffix='_ap').strip()
                        dims = inp.shape

                        if decl.startswith("hls::stream"):
                            if len(dims) == 1:
                                N = dims[0]
                                newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += (
                                    f'        ap_uint<{inp.type.name}::value_type::width> bits = ' f'temp[0].range();\n'
                                )
                                newline += f'        fout << bits.to_uint()' f' << (i+1<{N} ? \' \' : \'\\n\');\n'
                                newline += '    }\n'
                            else:
                                inputs_list = model.nn_config['inputs']
                                fifo_depth = next((e['fifo_depth'] for e in inputs_list if e['name'] == inp.name), None)
                                batch_size = next((e['batch_size'] for e in inputs_list if e['name'] == inp.name), None)
                                newline += f'    for(int r = 0; r < {fifo_depth}; r++) {{\n'
                                newline += f'        auto temp = {inp.name}_ap.read();\n'
                                newline += f'        for(int c = 0; c < {batch_size}; c++) {{\n'
                                newline += (
                                    f'            ap_uint<{inp.type.name}::value_type::width> bits = ' f'temp[c].range();\n'
                                )
                                newline += (
                                    f'            fout << bits.to_uint()' f' << (c+1<{batch_size} ? \' \' : \'\\n\');\n'
                                )
                                newline += '        }\n'
                                newline += '    }\n'
                        else:
                            ap = inp.name + "_ap"
                            N = inp.size_cpp()
                            newline += f'    for(int i = 0; i < {N}; i++) {{\n'
                            newline += f'        ap_uint<{inp.type.name}::width> bits = ' f'{ap}[i].range();\n'
                            newline += f'        fout << bits.to_uint()' f' << (i+1<{N} ? \' \' : \'\\n\');\n'
                            newline += '    }\n'
                    newline += "    fout.close();\n"
                    newline += "}\n"
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

    def write_build_script(self, model):
        """Write the TCL/Shell build scripts (build_prj.tcl, build_lib.sh, build_vra.sh, build_prj_bup.tcl, build_prj_bup.yml)

        Args:
            model (ModelGraph): the hls4ml model.
        """

        filedir = os.path.dirname(os.path.abspath(__file__))

        # Note: until we get a pragma hls_iterations, to insert ITERATIONS directive it will look like this:
        # directive set /myproject/nnet::conv_2d_cl<input_t,conv2d_1_result_t,config2>/core/nnet::conv_2d_window_cl<input_t,conv2d_1_result_t,config2>:do -ITERATIONS 64

        # build_prj.tcl
        srcpath = Path(filedir + '/../templates/catapult/build_prj.tcl').resolve()
        dstpath = Path(f'{model.config.get_output_dir()}/build_prj.tcl').resolve()
        with open(srcpath) as src, open(dstpath, 'w') as dst:
            for line in src.readlines():
                indent = line[: len(line) - len(line.lstrip())]
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('CATAPULT_DIR', model.config.get_project_dir())

                if '#hls-fpga-machine-learning insert build_options' in line:
                    bopts = model.config.get_config_value('BuildOptions')
                    line = 'array set BuildOptions {\n'
                    line = line + '  reset           0\n'
                    if not bopts is None:
                        for bopt in bopts:
                            if bopts[bopt] == '':
                                line = line + '  ' + bopt + ' {}\n'
                            else:
                                line = line + '  ' + bopt + ' ' + str(bopts[bopt]) + '\n'
                    else:
                        print('Warning - Catapult backend config was not created with create_initial_config')
                        line = line + '  csim            1\n'
                        line = line + '  SCVerify        0\n'
                        line = line + '  Synth           1\n'
                        line = line + '  vhdl            1\n'
                        line = line + '  verilog         1\n'
                        line = line + '  RTLSynth        0\n'
                        line = line + '  RandomTBFrames  2\n'
                        line = line + '  PowerEst        0\n'
                        line = line + '  PowerOpt        0\n'
                        line = line + '  BuildBUP        0\n'
                        line = line + '  BUPWorkers      0\n'
                        line = line + '  LaunchDA        0\n'
                        line = line + '  startup         {}\n'
                    line = line + '}\n'

                if '#hls-fpga-machine-learning insert techlibs' in line:
                    if model.config.get_config_value('Technology') is None:
                        if model.config.get_config_value('Part') is not None:
                            line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                        elif model.config.get_config_value('ASICLibs') is not None:
                            line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                    else:
                        if model.config.get_config_value('Technology') == 'asic':
                            line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                        else:
                            line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))

                elif '#hls-fpga-machine-learning insert invoke_args' in line:
                    tb_in_file = model.config.get_config_value('InputData')
                    tb_out_file = model.config.get_config_value('OutputPredictions')
                    invoke_args = '$sfd/firmware/weights'
                    if (tb_in_file is not None) and (tb_out_file is not None):
                        # regardless of the filepath in InputData or OutputPredictions, write_test_benches copied
                        # the file to build_prj.tcl/../tb_data/tb_input_features.dat etc
                        invoke_args = invoke_args + ' $sfd/tb_data/tb_input_features.dat'
                        invoke_args = invoke_args + ' $sfd/tb_data/tb_output_predictions.dat'
                        if model.config.get_config_value('CModelDefaultThreshold') is not None:
                            invoke_args = invoke_args + ' ' + str(model.config.get_config_value('CModelDefaultThreshold'))
                    line = indent + f'flow package option set /SCVerify/INVOKE_ARGS "{invoke_args}"\n'

                elif '#hls-fpga-machine-learning insert directives' in line:
                    results= re.search(r'.*#hls-fpga-machine-learning insert directives *(\w*)', line)
                    cat_dirs = model.config.get_config_value('GlobalDirectives')
                    if cat_dirs is not None:
                        for stage in cat_dirs:
                            if stage == results.group(1):
                                for cat_dir in cat_dirs[stage]:
                                    line = line + indent + f'directive set {cat_dir} {cat_dirs[stage][cat_dir]}\n'

                elif 'set hls_clock_period 5' in line:
                    line = indent + 'set hls_clock_period {}\n'.format(model.config.get_config_value('ClockPeriod'))
                dst.write(line)



        # build_prj_bup.tcl
        srcpath = Path(filedir + '/../templates/catapult/build_prj_bup.tcl').resolve()
        if os.path.exists(srcpath):
            dstpath = f'{model.config.get_output_dir()}/build_prj_bup.tcl'
            copyfile(srcpath, dstpath)

        # build_prj_bup.yml
        srcpath = Path(filedir + '/../templates/catapult/build_prj_bup.yml').resolve()
        dstpath = f'{model.config.get_output_dir()}/build_prj_bup.yml'
        if os.path.exists(srcpath):
            with open(srcpath) as src, open(dstpath, 'w') as dst:
                for line in src.readlines():
                    indent = line[: len(line) - len(line.lstrip())]
                    line = line.replace('myproject', model.config.get_project_name())
                    line = line.replace('CATAPULT_DIR', model.config.get_project_dir())

                    if '#hls-fpga-machine-learning insert build_options' in line:
                        line = ''
                        bopts = model.config.get_config_value('BuildOptions')
                        if not bopts is None:
                            for bopt in bopts:
                                if bopt == 'BuildBUP':
                                    line = line + indent + 'BuildBUP:        1\n'
                                else:
                                    if bopts[bopt] == '':
                                        line = line + indent + bopt + ': {}\n'
                                    else:
                                        line = line + indent + bopt + ': ' + str(bopts[bopt]) + '\n'
                        else:
                            print('Warning - Catapult backend config was not created with create_initial_config')
                            line = line + indent + 'csim:            1\n'
                            line = line + indent + 'SCVerify:        1\n'
                            line = line + indent + 'Synth:           1\n'
                            line = line + indent + 'vhdl:            1\n'
                            line = line + indent + 'verilog:         1\n'
                            line = line + indent + 'RTLSynth:        0\n'
                            line = line + indent + 'RandomTBFrames:  2\n'
                            line = line + indent + 'PowerEst:        0\n'
                            line = line + indent + 'PowerOpt:        0\n'
                            line = line + indent + 'BuildBUP:        1\n'
                            line = line + indent + 'BUPWorkers:      0\n'
                            line = line + indent + 'LaunchDA:        0\n'
                            line = line + indent + 'startup:         {}\n'

                    if '#hls-fpga-machine-learning insert techlibs' in line:
                        if model.config.get_config_value('Technology') is None:
                            if model.config.get_config_value('Part') is not None:
                                line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                            elif model.config.get_config_value('ASICLibs') is not None:
                                line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                        else:
                            if model.config.get_config_value('Technology') == 'asic':
                                line = indent + 'setup_asic_libs {{{}}}\n'.format(model.config.get_config_value('ASICLibs'))
                            else:
                                line = indent + 'setup_xilinx_part {{{}}}\n'.format(model.config.get_config_value('Part'))
                    elif '#hls-fpga-machine-learning insert invoke_args' in line:
                        # regardless of the filepath in InputData or OutputPredictions, write_test_benches copied
                        # the file to build_prj.tcl/../tb_data/tb_input_features.dat etc
                        tb_in_file = model.config.get_config_value('InputData')
                        tb_out_file = model.config.get_config_value('OutputPredictions')
                        invoke_args = '$sfd/firmware/weights'
                        if ((tb_in_file is not None) and (tb_out_file is not None)):
                            invoke_args = invoke_args + ' $sfd/tb_data/tb_input_features.dat'
                            invoke_args = invoke_args + ' $sfd/tb_data/tb_output_predictions.dat'
                            if model.config.get_config_value('CModelDefaultThreshold') is not None:
                                invoke_args = invoke_args + ' ' + str(model.config.get_config_value('CModelDefaultThreshold'))
                        line = indent + f'flow package option set /SCVerify/INVOKE_ARGS "{invoke_args}"\n'
                    elif '#hls-fpga-machine-learning insert directives' in line:
                        results= re.search(r'.*#hls-fpga-machine-learning insert directives *(\w*)', line)
                        cat_dirs = model.config.get_config_value('GlobalDirectives')
                        if cat_dirs is not None:
                            for stage in cat_dirs:
                                if stage == results.group(1):
                                    for cat_dir in cat_dirs[stage]:
                                        line = line + indent + f'directive set {cat_dir} {cat_dirs[stage][cat_dir]}\n'
                    elif 'set hls_clock_period 5' in line:
                        line = indent + 'set hls_clock_period {}\n'.format(model.config.get_config_value('ClockPeriod'))
                    dst.write(line)

        # build_lib.sh
        build_lib_src = Path(filedir + '/../templates/catapult/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

        # build_vra.sh
        build_lib_src = Path(filedir + '/../templates/catapult/build_vra.sh').resolve()
        if os.path.exists(build_lib_src):
            build_lib_dst = Path(f'{model.config.get_output_dir()}/build_vra.sh').resolve()
            with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
                for line in src.readlines():
                    line = line.replace('myproject', model.config.get_project_name())
                    line = line.replace('mystamp', model.config.get_config_value('Stamp'))
                    # regardless of the filepath in InputData or OutputPredictions, write_test_benches copied
                    # the file to build_prj.tcl/../tb_data/tb_input_features.dat etc
                    if model.config.get_config_value('InputData') is not None:
                        line = line.replace('tb_input_features.dat', 'tb_data/' + os.path.basename(model.config.get_config_value('InputData')))
                    if model.config.get_config_value('OutputPredictions') is not None:
                        line = line.replace('tb_output_predictions.dat', 'tb_data/' + os.path.basename(model.config.get_config_value('OutputPredictions')))
                    dst.write(line)
            build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)

    def write_build_script_multigraph(self, model):
        """Write the build script (build_lib.sh) for stitched multigraph project
        Args:
            model (MultiModelGraph): the hls4ml multigraph model.
        """
        # TODO - not yet implemented for Catapult
        ### filedir = Path(__file__).parent
        ### os.makedirs(model.config.get_output_dir(), exist_ok=True)
        ### build_lib_src = (filedir / '../templates/catapult/build_lib_multigraph.sh').resolve()
        ### build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        ### graph_project_names = ' '.join(f"\"{g.config.get_output_dir().split('/')[-1]}\"" for g in model.graphs)

        ### with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
        ###     for line in src.readlines():
        ###         line = line.replace('myproject', model.config.config['OriginalProjectName'])
        ###         line = line.replace('myproject_stitched', model.config.config['ProjectName'])
        ###         line = line.replace('mystamp', model.config.config['Stamp'])
        ###         line = line.replace('mygraph_name_list', graph_project_names)
        ###         dst.write(line)
        ### os.chmod(build_lib_dst, os.stat(build_lib_dst).st_mode | stat.S_IEXEC)

    def write_nnet_utils(self, model):
        """Copy the nnet_utils, AP types headers and any custom source to the project output directory

        Args:
            model (ModelGraph): the hls4ml model.
        """

        # nnet_utils
        filedir = os.path.dirname(os.path.abspath(__file__))
        if 'Mgc_home' in filedir:
            index = filedir.find('/Mgc_home/') + len('/Mgc_home/')
            MGC_HOME = filedir[0:index]
        else:
            MGC_HOME = os.getenv('MGC_HOME')

        dstpath = f'{model.config.get_output_dir()}/firmware/nnet_utils/'
        if not os.path.exists(dstpath):
            os.mkdir(dstpath)

        srcpath = os.path.join(filedir, '../templates/catapult/nnet_utils/')
        if not os.path.exists(srcpath+'nnet_types.h'):
            srcpath = os.path.join(MGC_HOME, 'shared/include/nnet_utils/')

        # nnet_utils
        copy_opt = model.config.get_config_value('CopyNNET')
        if (copy_opt is not None) and (copy_opt is True):
            headers = [os.path.basename(h) for h in glob.glob(srcpath + '*.h')]
            print("... copying NNET headers from " + srcpath)
            for h in headers:
                copyfile(srcpath + h, dstpath + h)
        else:
            print("... skipping copy of NNET headers from " + srcpath)
            h = 'nnet_code_gen.h'
            copyfile(srcpath + h, dstpath + h)

        # If the source for the AC packages is not found in the tree
        # where THIS python is located (e.g. this python is NOT in the
        # Catapult installation) then always copy the git submodules
        # for the AC packages into the output dir for this model.
        # Optionally, if the CopyAC option is set, then always copy
        # the AC package headers to the output dir for this model (but
        # rely on the bom files to specify the file copy list).
        copy_ac = model.config.get_config_value('CopyAC')
        filedir = os.path.dirname(os.path.abspath(__file__))
        for pkg in ('ac_types', 'ac_math', 'ac_simutils', 'ac_ipl'):
            dstpath = f'{model.config.get_output_dir()}/firmware/{pkg}/'

            # backward compatibility, look in root dir
            srcpath = os.path.join(filedir, '../../' + pkg + '/')
            if not os.path.exists(srcpath):
                # look next in Catapult-specific templates
                srcpath = os.path.join(filedir, '../templates/catapult/' + pkg + '/')

            if os.path.exists(srcpath):
                if os.path.exists(dstpath):
                    rmtree(dstpath)
                print("... copying AC " + pkg + " headers from " + srcpath)
                copytree(srcpath, dstpath)
            else:
                if (copy_ac is not None) and (copy_ac is True):
                    srcpath = f'{MGC_HOME}shared/'
                    print("... copying AC " + pkg + " headers from " + srcpath)
                    os.makedirs(dstpath,exist_ok=True)
                    os.makedirs(dstpath+'include',exist_ok=True)
                    os.makedirs(dstpath+'include/'+pkg,exist_ok=True)
                    # find $MGC_HOME/shared/pkgs/$pkg.any/rlsinfo/bom file
                    bom_file = f'{MGC_HOME}shared/pkgs/{pkg}/rlsinfo/bom'
                    with open(bom_file,'r') as bomf:
                        headers = bomf.read().splitlines()
                    for h in headers:
                        parent = os.path.dirname(dstpath + h.rstrip(os.sep))
                        if not os.path.exists(parent):
                            os.makedirs(parent,exist_ok=True)
                        #print(f'COPY {srcpath}{h} TO {dstpath}{h}')
                        copyfile(srcpath + h, dstpath + h)
                else:
                    print("... skipping copy of " + pkg + " headers - assumed to located in Catapult install tree")

        # custom source
        filedir = os.path.dirname(os.path.abspath(__file__))

        custom_source = get_backend('Catapult').get_custom_source()
        for dst, srcpath in custom_source.items():
            dstpath = f'{model.config.get_output_dir()}/firmware/{dst}'
            copyfile(srcpath, dstpath)

    def write_generated_code(self, model):
        """Write the generated code (nnet_code_gen.h)

        Args:
            model (ModelGraph): the hls4ml model.
        """
        path = f'{model.config.get_output_dir()}/firmware/nnet_utils/nnet_code_gen.h'
        f = open(path)
        contents = f.readlines()
        f.close()
        f = open(path, 'w')
        namespace = model.config.get_writer_config().get('Namespace', None)

        for line in contents:
            if '// hls4ml insert code' in line:
                newline = line
                for layer in model.get_layers():
                    for generated_code in layer.code.values():
                        newline += str(generated_code)
            else:
                newline = line
            if namespace is not None:
                if 'namespace nnet' in newline:
                    newline = newline.replace('namespace nnet', f'namespace {namespace}')
            f.write(newline)
        f.close()

    def write_yml(self, model):
        """Write the config to the YAML file

        Args:
            model (ModelGraph): the hls4ml model.
        """

        def keras_model_representer(dumper, keras_model):
            model_path = model.config.get_output_dir() + '/keras_model.keras'
            keras_model.save(model_path)
            return dumper.represent_scalar('!keras_model', model_path)

        try:
            from tensorflow.keras import Model as KerasModel

            yaml.add_multi_representer(KerasModel, keras_model_representer)
        except Exception:
            pass

        with open(model.config.get_output_dir() + '/' + config_filename, 'w') as file:
            yaml.dump(model.config.config, file)

    def write_tar(self, model):
        """Write the generated project as a .tar.gz archive

        Args:
            model (ModelGraph): the hls4ml model.
        """

        write_tar = model.config.get_writer_config().get('WriteTar', False)
        if write_tar:
            proposed_basename = model.config.get_output_dir()
            if proposed_basename == '.':
                proposed_basename = 'dot'
                print("Cannot write .tar.gz archive - output directory is '.'")
                return
            tar_path = proposed_basename + '.tar.gz'
            if os.path.exists(tar_path):
                os.remove(tar_path)
            with tarfile.open(tar_path, mode='w:gz') as archive:
                archive.add(model.config.get_output_dir(), recursive=True)

    def write_layer_summary(self, model):

        df = pd.DataFrame()
        input_shape = ""
        input_datatype = ""


        def get_weight_typedef(cpp: str):
            m = re.match(r'^\s*typedef\s+(.*?)\s+(\w+)\s*;\s*$', cpp, flags=re.DOTALL)
            if not m:
                return None
            raw_type, alias = m.group(1).strip(), m.group(2).lower()
            if "weight" not in alias:
                return None

            # Check if PO2 struct type
            struct_match = re.match(r'struct\s+\w+\s*{([\s\S]*?)}', raw_type, flags=re.DOTALL)
            if struct_match:
                body = struct_match.group(1)

                # Extract exponent type (the type of field named 'weight')
                exp = re.search(r'\b(\w+<[^>]+>)\s+weight\b', body)
                if exp:
                    exp_type = exp.group(1).strip()
                    return f"POW2(Sign+{exp_type})"

                return None

            # Otherwise normal ac_fixed typedef
            return raw_type


        def get_bias_typedef(cpp: str):
            m = re.match(r'^\s*typedef\s+(.*?)\s+(\w+)\s*;\s*$', cpp, flags=re.DOTALL)
            if not m:
                return None
            raw_type, alias = m.group(1).strip(), m.group(2).lower()
            if "bias" not in alias:
                return None

            # Check if PO2 struct type
            struct_match = re.match(r'struct\s+\w+\s*{([\s\S]*?)}', raw_type, flags=re.DOTALL)
            if struct_match:
                body = struct_match.group(1)

                # Extract exponent type from `weight` field
                exp = re.search(r'\b(\w+<[^>]+>)\s+weight\b', body)
                if exp:
                    exp_type = exp.group(1).strip()
                    return f"POW2(Sign+{exp_type})"

                return None

            # Otherwise normal ac_fixed typedef
            return raw_type


        for layer in model.get_layers():
            # Output datatype string
            datatype = layer.get_output_variable().type.precision.definition_cpp() + " "
            all_precision = OrderedDict()
            layer_precision = layer.get_layer_precision()
            for type_name, type_var in layer_precision.items():
                        # Ensure that layer's types don't override existing types
                        # (InplaceVariable types can appear multiple times)
                        if type_name not in all_precision:
                            all_precision[type_name] = type_var

            # Output shape
            shape = ""
            for v in layer.get_output_variable().get_shape():
                shape += "[" + str(v) + "]"

            if layer.attributes.layer.class_name != 'Input':
                my_class_name = layer.class_name
                if layer.attributes.layer.class_name == 'Activation':
                    my_class_name = layer.get_attr('activation')

                # Filter size
                filter = ""
                filt_width = layer.get_attr('filt_width')
                filt_height = layer.get_attr('filt_height')
                if filt_width is not None:
                    filter = "[" + str(filt_width) + "]"
                if filt_height is not None:
                    filter += "[" + str(filt_height) + "]"

                # Stride
                stride = ""
                stride_width = layer.get_attr('stride_width')
                if stride_width is not None:
                    stride = str(stride_width)

                # Detect weight and bias types
                weight_type = ''
                bias_type = ''

                for used_type in all_precision.values():
                    cpp = used_type.definition_cpp()

                    if (wt := get_weight_typedef(cpp)):
                        weight_type = wt   # assign only if actual weight type

                    if (bt := get_bias_typedef(cpp)):
                        bias_type = bt     # assign only if actual bias type

                new_row = {
                    'Layer Name': [layer.name],
                    'Layer Class': [my_class_name],
                    'Input Type': [input_datatype],
                    'Input Shape': [input_shape],
                    'Output Type': [datatype],
                    'Output Shape': [shape],
                    'Weight Type': [weight_type],
                    'Bias Type': [bias_type],
                    'Filter Shape': [filter],
                    'Stride': [stride],
                    'Reuse': [layer.model.config.get_reuse_factor(layer)]
                }
                df_row = pd.DataFrame(new_row)
                df = pd.concat([df, df_row], ignore_index=True)

            input_shape = shape
            input_datatype = datatype

        df.to_csv(f'{model.config.get_output_dir()}/firmware/layer_summary.csv', sep=';', index=False)

        with open(f'{model.config.get_output_dir()}/firmware/layer_summary.txt', 'w') as rptfile:
            print(df.to_markdown(index=False, tablefmt='simple', floatfmt='.8f',
                                colalign=['center'] * len(df.columns)), file=rptfile)

    def write_hls(self, model):
        
        memory_type = (
            model.config.get_config_value('MemType') if model.config.get_config_value('MemType') is not None else None
        )
        port_type = (
            model.config.get_config_value('PortType') if model.config.get_config_value('PortType') is not None else None
        )
        if memory_type == 'RAM':
            print('RAM not supported')

        print('Writing HLS project')
        self.write_output_dir(model)
        self.write_project_cpp(model, memory_type, port_type)
        self.write_project_header(model, memory_type, port_type)
        self.write_layer_summary(model)
        self.write_weights(model)
        self.write_defines(model)
        self.write_parameters(model)
        self.write_test_bench(model, memory_type, port_type)
        self.write_bridge(model, memory_type, port_type)
        self.write_build_script(model)
        self.write_nnet_utils(model)
        self.write_generated_code(model)
        self.write_yml(model)
        self.write_tar(model)
        print('Done')
