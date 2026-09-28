#ifndef NNET_CONV2D_STREAM_H_
#define NNET_CONV2D_STREAM_H_

#include "ap_shift_reg.h"
#include "nnet_common.h"
#include <ac_ipl/ac_window_v2_flush.h>

#include "nnet_conv_stream.h"
// #include "nnet_conv_acwin.h"
#include "nnet_con2d_win_class.h"

#include <ac_channel.h>
#include <ac_sync.h>
// #include <ac_ipl/ac_window_v2.h>
#include <assert.h>

namespace nnet {

template <class data_T, typename CONFIG_T>
void compute_scaled_indices_2d(const unsigned h_idx, const unsigned w_idx,
                               ac_int<CONFIG_T::filt_height * CONFIG_T::filt_width, false> *pixel_idx) 
{
    const unsigned sh_idx = CONFIG_T::template scale_index_height<CONFIG_T::filt_height, CONFIG_T::stride_height,
                                                                  CONFIG_T::in_height>::scale_index(h_idx);
    unsigned wp_idx = w_idx * (data_T::size / CONFIG_T::n_chan);

    #pragma hls_unroll
    ComputeIndex: for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
        unsigned sw_idx = CONFIG_T::template scale_index_width<CONFIG_T::filt_width, CONFIG_T::stride_width,
                                                               CONFIG_T::in_width>::scale_index(wp_idx + p);
        pixel_idx[p] = CONFIG_T::pixels[sh_idx * CONFIG_T::min_width + sw_idx];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_encoded_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == CONFIG_T::filt_width);

    ac_channel<typename data_T::value_type> data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];

    res_T res_pack;
    unsigned outputs_ready = 0;

    ac_int<CONFIG_T::filt_height * CONFIG_T::filt_width, false> pixel_idx[data_T::size / CONFIG_T::n_chan];

    constexpr int ce_reuse_factor =
        CONFIG_T::reuse_factor * ((CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic) && data_T::size / CONFIG_T::n_chan == 1);
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval 1
    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            compute_scaled_indices_2d<data_T, CONFIG_T>(i_ih, i_iw, pixel_idx);
            compute_output_encoded<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready, weights,
                                                            biases, pixel_idx);
        }
    }
}

// Line Buffer
template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_buffer_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[MAX(CONFIG_T::filt_height - 1, 1)]
                                                                                    [CONFIG_T::n_chan];

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor * (CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic);
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval 1
    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            if (CONFIG_T::filt_height > 1) {
                compute_output_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res, weights, biases);
            } else {
                compute_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            }
        }
    }
}

#pragma hls_design
#pragma hls_resource weights.weight:rsc variable="weights.weight" map_to_module="[DirectInput]"
#pragma hls_resource weights.sign:rsc variable="weights.sign" map_to_module="[DirectInput]"
#pragma hls_resource biases.weight:rsc variable="biases.weight" map_to_module="[DirectInput]"
#pragma hls_resource biases.sign:rsc variable="biases.sign" map_to_module="[DirectInput]"
#pragma hls_resource weights:rsc variable="weights" map_to_module="[DirectInput]"
#pragma hls_resource biases:rsc variable="biases" map_to_module="[DirectInput]"
template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt], ac_sync &sync_w, ac_sync &sync_b) 
{
    sync_w.sync_in(weights);
    sync_b.sync_in(biases);

    if constexpr(CONFIG_T::implementation == conv_implementation::linebuffer) {
        conv_2d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
    if constexpr(CONFIG_T::implementation == conv_implementation::encoded) {
        conv_2d_encoded_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
    if constexpr(CONFIG_T::implementation == conv_implementation::ac_window) {
        conv2d_window_v2_dispatch<data_T, res_T, CONFIG_T, CONFIG_T::padding>::run(data, res, weights, biases);
    }
}

// --------------------
// ac_window top
// --------------------
#pragma hls_design block
#pragma hls_pipeline_init_interval 1
template <class data_T, class res_T, typename CONFIG_T,
          typename std::enable_if<(CONFIG_T::implementation == conv_implementation::ac_window), int>::type = 0>
void conv_2d_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width *
                                        CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    conv2d_window_v2_dispatch<data_T, res_T, CONFIG_T, CONFIG_T::padding>::run(
        data, res, weights, biases);
}

// --------------------
// non-ac_window top
// --------------------
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T,
          typename std::enable_if<(CONFIG_T::implementation != conv_implementation::ac_window), int>::type = 0>
void conv_2d_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width *
                                        CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    if constexpr (CONFIG_T::implementation == conv_implementation::linebuffer) {
        conv_2d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if constexpr (CONFIG_T::implementation == conv_implementation::encoded) {
        conv_2d_encoded_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet
#endif

