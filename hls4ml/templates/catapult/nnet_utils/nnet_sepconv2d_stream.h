#ifndef NNET_SEPARABLE_CONV2D_STREAM_H_
#define NNET_SEPARABLE_CONV2D_STREAM_H_

#include "nnet_common.h"
#include <ac_ipl/ac_window_v2_flush.h>

#include "nnet_conv2d_stream.h"
#include "nnet_sepconv_stream.h"
#include "nnet_sep_conv2d_win_class.h"
// #include "nnet_conv_acwin.h"

#include "nnet_types.h"
#include <ac_channel.h>
#include <assert.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_encoded_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) 
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == CONFIG_T::filt_width);

    static ac_channel<typename data_T::value_type>
        data_window[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];

    res_T res_pack;
    unsigned outputs_ready = 0;
    ac_int<CONFIG_T::filt_height * CONFIG_T::filt_width, false> pixel_idx[data_T::size / CONFIG_T::n_chan];
    constexpr int ce_reuse_factor =
        CONFIG_T::reuse_factor * ((CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic) && data_T::size / CONFIG_T::n_chan == 1);
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            compute_scaled_indices_2d<data_T, CONFIG_T>(i_ih, i_iw, pixel_idx);
            compute_depthwise_output_encoded<data_T, res_T, CONFIG_T>(data.read(), data_window, res, res_pack, outputs_ready,
                                                                      weights, biases, pixel_idx);
        }
    }
}

// Line Buffer Implementation (Phil's)
template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_buffer_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                                 typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_filt],
                                 typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width> line_buffer[CONFIG_T::filt_height - 1]
                                                                                    [CONFIG_T::n_chan];

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor * (CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic);
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
            if (CONFIG_T::filt_height > 1) {
                compute_depthwise_output_buffer_2d<data_T, res_T, CONFIG_T>(data.read(), line_buffer, res, weights, biases);
            } else {
                compute_depthwise_output_buffer_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            }
        }
    }
}

// --------------------
// ac_window top (depthwise)
// --------------------
#pragma hls_design block
#pragma hls_pipeline_init_interval 1
template <class data_T, class res_T, typename CONFIG_T,
          typename std::enable_if<(CONFIG_T::implementation == conv_implementation::ac_window), int>::type = 0>
void depthwise_conv_2d_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    depthwise_conv_2d_window_dispatch_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

// --------------------
// non-ac_window top (depthwise)
// --------------------
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T,
          typename std::enable_if<(CONFIG_T::implementation != conv_implementation::ac_window), int>::type = 0>
void depthwise_conv_2d_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    if constexpr (CONFIG_T::implementation == conv_implementation::linebuffer) {
        depthwise_conv_2d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else if constexpr (CONFIG_T::implementation == conv_implementation::encoded) {
        depthwise_conv_2d_encoded_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}


template <class data_T, class res_T, typename CONFIG_T>
void pointwise_linebuffer(ac_channel<data_T> &data, ac_channel<res_T> &res,
                                 typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                                 typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{

    assert(CONFIG_T::pad_top == 0 && CONFIG_T::pad_bottom == 0 && CONFIG_T::pad_left == 0 && CONFIG_T::pad_right == 0);
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    constexpr int ce_reuse_factor =
        CONFIG_T::reuse_factor * ((CONFIG_T::strategy == nnet::latency || CONFIG_T::strategy == nnet::distributed_arithmetic) && data_T::size / CONFIG_T::n_chan == 1);
    (void)ce_reuse_factor;

    #pragma hls_pipeline_init_interval ce_reuse_factor
    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / (data_T::size / CONFIG_T::n_chan); i_iw++) {
            if (i_ih % CONFIG_T::stride_height == 0 && i_iw % CONFIG_T::stride_width == 0) {
                pointwise_mult_buffer<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
            } else {
                data.read(); // discard input
            }
        }
    }
}

// ------------------------------
// AC_WINDOW specialization
// ------------------------------
#pragma hls_design block
#pragma hls_pipeline_init_interval 1
template <class data_T, class res_T, typename CONFIG_T,
          typename std::enable_if<(CONFIG_T::implementation == conv_implementation::ac_window), int>::type = 0>
void pointwise_conv_2d_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt])
{
    // AC_WINDOW body directly here (no separate function name)
    constexpr int BUS_WORDS = data_T::dim1;

    data_T in_array;
    typedef typename data_T::ElemType in_vector_t;
    in_vector_t in_vector;
    constexpr int N_CHANNELS_IN = in_vector_t::packed_words;

    typedef typename res_T::ElemType out_vector_t;
    out_vector_t res_vector;
    res_T out_vector;
    constexpr int N_CHANNELS_OUT = out_vector_t::packed_words;

    typedef typename in_vector_t::base_type  in_base_t;
    typedef typename out_vector_t::base_type out_base_t;

    in_base_t  ch_data[N_CHANNELS_IN];
    out_base_t ch_res[N_CHANNELS_OUT];

    #pragma hls_pipeline_init_interval 1
    ReadInputHeight_win: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth_win: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / BUS_WORDS; i_iw++) {
            in_array = data.read();

            #pragma hls_unroll yes
            ReadInputWord_win: for (unsigned bw = 0; bw < BUS_WORDS; bw++) {
                in_vector = in_array[bw];

                #pragma hls_unroll yes
                ReadChannel_win: for (unsigned ch = 0; ch < N_CHANNELS_IN; ch++) {
                    ch_data[ch] = in_vector[ch];
                }

                dense_latency<in_base_t, out_base_t, typename CONFIG_T::mult_config, nnet::II_RF>(
                    ch_data, ch_res, weights, biases);

                #pragma hls_unroll yes
                WriteChannel_win: for (unsigned ch = 0; ch < N_CHANNELS_OUT; ch++) {
                    res_vector[ch] = ch_res[ch];
                }

                out_vector[bw] = res_vector;
            }
            res.write(out_vector);
        }
    }
}

// ------------------------------
// LINEBUFFER specialization
// ------------------------------
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T,
          typename std::enable_if<(CONFIG_T::implementation != conv_implementation::ac_window), int>::type = 0>
void pointwise_conv_2d_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt])
{
    // Call existing linebuffer implementation
    pointwise_linebuffer<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

#pragma hls_design block
template <class data_T, class dw_res_T, class res_T, typename CONFIG_T>
void separable_conv_2d_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                          typename CONFIG_T::depthwise_config::weight_t
                              depthwise_weights[CONFIG_T::depthwise_config::filt_height *
                                                CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_filt],
                          typename CONFIG_T::pointwise_config::weight_t
                              pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
                          typename CONFIG_T::depthwise_config::bias_t depthwise_biases[CONFIG_T::depthwise_config::n_filt],
                          typename CONFIG_T::pointwise_config::bias_t pointwise_biases[CONFIG_T::pointwise_config::n_filt]) 
{
    static ac_channel<dw_res_T> depthwise_res;
    constexpr unsigned res_depth = CONFIG_T::depthwise_config::out_height * CONFIG_T::depthwise_config::out_width;

    depthwise_conv_2d_cl<data_T, dw_res_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights,
                                                                                depthwise_biases);
    pointwise_conv_2d_cl<dw_res_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights,
                                                                               pointwise_biases);
}

} // namespace nnet
#endif

