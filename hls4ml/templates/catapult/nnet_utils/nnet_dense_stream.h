#ifndef NNET_DENSE_STREAM_H_
#define NNET_DENSE_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"
#include <ac_channel.h>
#include <ac_sync.h>
#include <assert.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_wrapper(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) 
{
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor * 
        (CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic);
    (void)ce_reuse_factor;

    #pragma hls_pipeline_init_interval ce_reuse_factor
    CONFIG_T::template kernel<data_T, res_T, CONFIG_T>::dense(data, res, weights, biases);
}

#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <class data_T, class res_T, typename CONFIG_T>
void dense(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
           typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
           typename CONFIG_T::bias_t biases[CONFIG_T::n_out], ac_sync &sync_w, ac_sync &sync_b) 
{
    typename data_T::value_type data[CONFIG_T::n_in];
    typename res_T::value_type res[CONFIG_T::n_out];

    sync_w.sync_in(weights);
    sync_b.sync_in(biases);

    if ((CONFIG_T::n_in / data_T::size) > 1) {
        #pragma hls_pipeline_init_interval 1
    }
    DataPrepare: for (unsigned int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
        #pragma hls_unroll
        DataPack: for (unsigned int i_pack = 0; i_pack < data_T::size; i_pack++) {
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

    if ((CONFIG_T::n_out / res_T::size) > 1) {
        #pragma hls_pipeline_init_interval 1
    }
    ResWrite: for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        res_T res_pack;
        #pragma hls_unroll
        ResPack: for (unsigned int i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

#pragma hls_design
#pragma hls_pipeline_init_interval 1
template <class data_T, class res_T, typename CONFIG_T>
void dense(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream,
           typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
           typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) 
{
    typename data_T::value_type data[CONFIG_T::n_in];

    typename res_T::value_type res[CONFIG_T::n_out];

    if ((CONFIG_T::n_in / data_T::size) > 1) {
        #pragma hls_pipeline_init_interval 1
    }
    DataPrepare: for (unsigned int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
        #pragma hls_unroll
        DataPack: for (unsigned int i_pack = 0; i_pack < data_T::size; i_pack++) {
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    dense_wrapper<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);

    if ((CONFIG_T::n_out / res_T::size) > 1) {
        #pragma hls_pipeline_init_interval 1
    }
    ResWrite: for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        res_T res_pack;
        #pragma hls_unroll
        ResPack: for (unsigned int i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

} // namespace nnet

#endif

