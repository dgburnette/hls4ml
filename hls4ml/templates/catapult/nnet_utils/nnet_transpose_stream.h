#ifndef NNET_TRANSPOSE_STREAM_H
#define NNET_TRANSPOSE_STREAM_H

#include "nnet_transpose.h"
#include <ac_channel.h>
#include <type_traits>

namespace nnet {

template <typename data_T, typename res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::dims == 2, void>::type transpose(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    typename data_T::value_type data_array[CONFIG_T::N];

    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < CONFIG_T::N / data_T::size; i++) {
        #pragma hls_unroll
        data_T in_data = data.read();
        for (int j = 0; j < data_T::size; j++) {
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < CONFIG_T::N / res_T::size; i++) {
        res_T out_data;
        #pragma hls_unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = typename res_T::value_type(data_array[j * CONFIG_T::from_shape[1] + i]);
        }
        res.write(out_data);
    }
}

// This sfinae is for vivado_hls, which has some overhead using the transfer_idx in io_stream.
// In vitis both performs exactly the same, thus this is not removed out of convenience.
template <typename data_T, typename res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::dims != 2, void>::type transpose(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    typename data_T::value_type data_array[CONFIG_T::N];

    #pragma hls_pipeline_init_interval 1
    for (unsigned int i = 0; i < CONFIG_T::N / data_T::size; i++) {
        data_T in_data = data.read();
        #pragma hls_unroll
        for (unsigned int j = 0; j < data_T::size; j++) {
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    #pragma hls_pipeline_init_interval 1
    for (unsigned int i = 0; i < CONFIG_T::N / res_T::size; i++) {
        res_T out_data;
        #pragma hls_unroll
        for (unsigned int j = 0; j < res_T::size; j++) {
            out_data[j] = typename res_T::value_type(data_array[transfer_idx<CONFIG_T>(i * res_T::size + j)]);
        }
        res.write(out_data);
    }
}

} // namespace nnet
#endif

