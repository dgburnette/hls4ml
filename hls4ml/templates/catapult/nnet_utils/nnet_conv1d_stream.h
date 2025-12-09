#ifndef NNET_CONV1D_STREAM_H_
#define NNET_CONV1D_STREAM_H_

#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include <ac_channel.h>

namespace nnet {

template <class data_T, typename CONFIG_T>
void compute_scaled_indices_1d(const unsigned w_idx, ac_int<CONFIG_T::filt_width, false> *pixel_idx) 
{
    unsigned wp_idx = w_idx * (data_T::size / CONFIG_T::n_chan);

    #pragma hls_unroll
    ComputeIndex: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        unsigned sw_idx =
            CONFIG_T::template scale_index<CONFIG_T::filt_width, CONFIG_T::stride_width, CONFIG_T::in_width>::scale_index(
                wp_idx + p);
        pixel_idx[p] = CONFIG_T::pixels[sw_idx];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_encoded_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                        typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    ac_channel<typename data_T::value_type> data_window[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_pack;
    unsigned outputs_ready = 0;

    ac_int<CONFIG_T::filt_width, false> pixel_idx[data_T::size / CONFIG_T::n_chan];

    constexpr int ce_reuse_factor =
        CONFIG_T::reuse_factor * ((CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic) &&
            data_T::size / CONFIG_T::n_chan == 1);
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
        compute_scaled_indices_1d<data_T, CONFIG_T>(i_iw, pixel_idx);
        compute_output_encoded<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready, weights,
                                                        biases, pixel_idx);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_buffer_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                       typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                       typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    assert(CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor * (CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic);
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
        compute_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
    }
}

#pragma hls_design
template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    switch (CONFIG_T::implementation) {
    case conv_implementation::linebuffer:
        conv_1d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
        break;
    case conv_implementation::encoded:
        conv_1d_encoded_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
        break;
    }
}

} // namespace nnet
#endif

