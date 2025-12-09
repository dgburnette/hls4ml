#ifndef NNET_SEPARABLE_CONV2D_STREAM_H_
#define NNET_SEPARABLE_CONV2D_STREAM_H_

#include "nnet_common.h"
#include "nnet_conv2d_stream.h"
#include "nnet_sepconv_stream.h"
#include "nnet_types.h"
#include <ac_channel.h>
#include <ac_ipl/ac_window_v2.h>
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

// AC_WINDOW V2 BASED IMPLEMENTATION
template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_window_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                                 typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
                                 typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor * (CONFIG_T::strategy == nnet::latency);
    (void)ce_reuse_factor;

    enum {
        AC_IMG_HEIGHT = CONFIG_T::in_height,
        AC_IMG_WIDTH = CONFIG_T::in_width,
        AC_WIN_HEIGHT = CONFIG_T::filt_height,
        AC_WIN_WIDTH = CONFIG_T::filt_width,
    };
    assert(!(CONFIG_T::padding == padding_type::valid && AC_BUS_WORDS > AC_WIN_WIDTH) && "ac_window with 'valid' padding does not support AC_BUS_WORDS greater than AC_WIN_WIDTH.");
    assert(!(CONFIG_T::padding == padding_type::valid && ((AC_WIN_WIDTH - 1) % AC_BUS_WORDS != 0)) && "ac_window with 'valid' padding requires (AC_WIN_WIDTH - 1) to be divisible by AC_BUS_WORDS.");

    ac_int<ac::nbits<CONFIG_T::in_width>::val, false> width = CONFIG_T::in_width;
    ac_int<ac::nbits<CONFIG_T::in_height>::val, false> height = CONFIG_T::in_height;

    constexpr ac_buff_type BUFF_TYPE = (CONFIG_T::in_width % 2 == 0) ? AC_SPWRMASK : AC_DUAL;
    constexpr ac_padding_method AC_PMODE = CONFIG_T::padding == padding_type::same ? AC_CONSTANT : AC_NO_PADDING;
    typedef typename data_T::ElemType in_vector_t;
    typedef typename res_T::ElemType out_vector_t;
    typedef typename in_vector_t::base_type in_base_t;
    typedef typename out_vector_t::base_type out_base_t;
    typename CONFIG_T::accum_t acc[CONFIG_T::n_filt];
    ac_array<in_vector_t, CONFIG_T::filt_height, CONFIG_T::filt_width> single_window;
    static typename in_vector_t::base_type kernel_data[CONFIG_T::kernel_size * CONFIG_T::n_chan];
    typename out_vector_t::base_type res_out[CONFIG_T::n_filt];
    out_vector_t out_vector;
    res_T res_pack;
    constexpr int d_mult=CONFIG_T::n_filt/CONFIG_T::n_chan;
    typedef ac_window_v2_2d<in_vector_t, AC_IMG_HEIGHT, AC_IMG_WIDTH, AC_WIN_HEIGHT, AC_WIN_WIDTH, BUFF_TYPE, AC_PMODE, AC_BUS_WORDS> WINDOW_2D_TYPE;

    bool eof_out = false, in_read = true;
    WINDOW_2D_TYPE window;
     
constexpr int iterations = CONFIG_T::padding == padding_type::same
    ? ((AC_IMG_WIDTH * AC_IMG_HEIGHT) + (((AC_WIN_HEIGHT / 2) - (AC_WIN_HEIGHT % 2 == 0 ? 1 : 0)) * AC_IMG_WIDTH + ((AC_WIN_WIDTH / 2) - (AC_WIN_WIDTH % 2 == 0 ? 1 : 0)))) / AC_BUS_WORDS
    : (AC_IMG_WIDTH * AC_IMG_HEIGHT) / AC_BUS_WORDS;

    (void) iterations; // to prevent compiler warnings
    #pragma hls_iterations iterations
    #pragma hls_pipeline_init_interval ce_reuse_factor
    do {
        data_T data_dummy;
        data_T din = in_read ? data.read() : data_dummy;
        ac_array<data_T, AC_BUS_WORDS> din_array;
        // din_array[0] = din;   //Check this do we need a array or one element 

        constexpr int AC_WORDS = WINDOW_2D_TYPE::AC_WORDS;
        ac_array<in_vector_t, CONFIG_T::filt_height, AC_WORDS> window_out;
        bool sof_out, sol_out, eol_out;
        ac_int<AC_BUS_WORDS, false> vld_out;
        window.run(din, width, height, in_read, window_out, sof_out, eof_out, sol_out, eol_out, vld_out);

        if (vld_out) {
            #pragma hls_unroll yes
            AcBusNum: for (int m = 0; m < AC_BUS_WORDS; m++) {
                #pragma hls_unroll yes
                HeightIndex: for (unsigned i = 0; i < CONFIG_T::filt_height; i++) {
                    #pragma hls_unroll yes
                    WidthIndex: for (unsigned j = 0; j < CONFIG_T::filt_width; j++) {
                        single_window[i][j] = window_out[i][j + m];
                        in_vector_t in_elem=single_window[i][j];
                        #pragma hls_unroll yes
                        ChannelIndex: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                            kernel_data[i * CONFIG_T::filt_width * CONFIG_T::n_chan + j * CONFIG_T::n_chan + k] = in_base_t(in_elem[k]);
                        }
                    }
                }
                #pragma hls_unroll
                    DepthwiseLoop: for (unsigned int jj = 0; jj < CONFIG_T::n_chan; jj++) {
                        #pragma hls_unroll
                        for (unsigned int kk = 0; kk < d_mult; kk++) {
                            int filt_idx = jj * d_mult + kk;
                            acc[filt_idx] = (typename CONFIG_T::accum_t)biases[filt_idx];
                            #pragma hls_unroll
                            for (unsigned int ii = 0; ii < CONFIG_T::kernel_size; ii++) {
                                int data_idx = ii * CONFIG_T::n_chan + jj;
                                int weight_idx = (ii * CONFIG_T::n_chan + jj) * d_mult + kk;
                                acc[filt_idx] += CONFIG_T::mult_config::template product<
                                    in_base_t, typename CONFIG_T::mult_config::weight_t>::product(
                                        kernel_data[data_idx], weights[weight_idx]);
                            }
                            res_out[filt_idx] = cast<in_base_t, out_base_t, typename CONFIG_T::mult_config>(acc[filt_idx]);
                        }
                    }
                #pragma hls_unroll yes
                CastLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
                    out_vector[i_ic] = res_out[i_ic];
                }
                res_pack[m]=out_vector;
            }
            res.write(res_pack);
        }
    } while(!eof_out);
}



#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_filt],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{ 
    //TODO assert((CONFIG_T::n_filt == CONFIG_T::n_chan) && "only a depth multiplier of 1 is currently supported");

    //To ensure only the required implementation methods compile.
    if constexpr(CONFIG_T::implementation == conv_implementation::linebuffer) {
        depthwise_conv_2d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
    if constexpr(CONFIG_T::implementation == conv_implementation::encoded) {
        depthwise_conv_2d_encoded_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
    if constexpr(CONFIG_T::implementation == conv_implementation::ac_window) {
        depthwise_conv_2d_window_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
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

    #pragma hls_pipeline_init_interval 1
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

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_ac_window(ac_channel<data_T> &data, ac_channel<res_T> &res,
                         typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                         typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    // std::cout << "INFO: ac_window path selected\n";

    constexpr int BUS_WORDS = data_T::dim1;
    data_T in_array;
    typedef typename data_T::ElemType in_vector_t;
    in_vector_t in_vector;
    constexpr int N_CHANNELS_IN = in_vector_t::packed_words;

    typedef typename res_T::ElemType out_vector_t;
    out_vector_t res_vector;
    res_T out_vector;
    constexpr int N_CHANNELS_OUT = out_vector_t::packed_words;

    typedef typename in_vector_t::base_type in_base_t;
    typedef typename out_vector_t::base_type out_base_t;

    in_base_t ch_data[N_CHANNELS_IN];
    out_base_t ch_res[N_CHANNELS_OUT];

    #pragma hls_pipeline_init_interval 1
    ReadInputHeight: for (unsigned i_ih = 0; i_ih < CONFIG_T::in_height; i_ih++) {
        ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width / BUS_WORDS; i_iw++) {
            in_array = data.read();
            
            #pragma hls_unroll yes
            ReadInputWord: for (unsigned bw = 0; bw < BUS_WORDS; bw++) {
                in_vector = in_array[bw];
               
                #pragma hls_unroll yes
                ReadChannel: for (unsigned ch = 0; ch < N_CHANNELS_IN; ch++) {
                    ch_data[ch] = in_vector[ch];
                }

                dense_latency<in_base_t, out_base_t, typename CONFIG_T::mult_config, nnet::II_RF>(
                    ch_data, ch_res, weights, biases);
    
                #pragma hls_unroll yes
                WriteChannel: for (unsigned ch = 0; ch < N_CHANNELS_OUT; ch++) {
                    res_vector[ch] = ch_res[ch];
                }

                out_vector[bw] = res_vector;
            }
            res.write(out_vector);
        }
    }
}

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_cl(ac_channel<data_T> &data, ac_channel<res_T> &res,
                          typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                          typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{


    if constexpr (CONFIG_T::implementation == conv_implementation::ac_window) {
        pointwise_ac_window<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        pointwise_linebuffer<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
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

