#ifndef NNET_CONV2D_STREAM_H_
#define NNET_CONV2D_STREAM_H_

#include "ap_shift_reg.h"
#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include <ac_channel.h>
#include <ac_sync.h>
#include <ac_ipl/ac_window_v2.h>
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

// AC_WINDOW_V2 Library
template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_window_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
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
    assert((AC_BUS_WORDS <= AC_IMG_WIDTH) && "ac_window implementation does not support AC_BUS_WORDS greater than feature size (AC_IMG_WIDTH).");
    assert(!(CONFIG_T::padding == padding_type::valid && AC_BUS_WORDS > AC_WIN_WIDTH) && "ac_window with 'valid' padding does not support AC_BUS_WORDS greater than AC_WIN_WIDTH.");
    assert(!(CONFIG_T::padding == padding_type::valid && ((AC_WIN_WIDTH - 1) % AC_BUS_WORDS != 0)) && "ac_window with 'valid' padding requires (AC_WIN_WIDTH - 1) to be divisible by AC_BUS_WORDS.");

    ac_int<ac::nbits<CONFIG_T::in_width>::val, false> width = CONFIG_T::in_width;
    ac_int<ac::nbits<CONFIG_T::in_height>::val, false> height = CONFIG_T::in_height;

    typedef typename data_T::ElemType in_vector_t;
    typedef typename res_T::ElemType out_vector_t;
    typedef typename in_vector_t::base_type in_base_t;
    typedef typename out_vector_t::base_type out_base_t;

    ac_array<in_vector_t, CONFIG_T::filt_height, CONFIG_T::filt_width> single_window;
    static typename in_vector_t::base_type kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    typename out_vector_t::base_type res_out[CONFIG_T::n_filt];
    out_vector_t out_vector;
    res_T res_pack;

    constexpr ac_buff_type BUFF_TYPE = (CONFIG_T::in_width % 2 == 0) ? AC_SPWRMASK : AC_DUAL;
    constexpr ac_padding_method AC_PMODE = CONFIG_T::padding == padding_type::same ? AC_CONSTANT : AC_NO_PADDING;
    typedef ac_window_v2_2d<in_vector_t, AC_IMG_HEIGHT, AC_IMG_WIDTH, AC_WIN_HEIGHT, AC_WIN_WIDTH, BUFF_TYPE, AC_PMODE, AC_BUS_WORDS> WINDOW_2D_TYPE;

    bool eof_out = false, in_read = true;
    WINDOW_2D_TYPE window;

    constexpr int iterations = CONFIG_T::padding == padding_type::same
        ? (AC_IMG_WIDTH*AC_IMG_HEIGHT)+(((AC_WIN_HEIGHT / 2)-(AC_WIN_HEIGHT % 2 == 0))*AC_IMG_WIDTH+((AC_WIN_WIDTH/2)-(AC_WIN_WIDTH % 2 == 0)))
        : (AC_IMG_WIDTH*AC_IMG_HEIGHT);
    (void) iterations; // to prevent compiler warnings
    #pragma hls_iterations iterations
    #pragma hls_pipeline_init_interval ce_reuse_factor
    do {
        data_T data_dummy;
        data_T din = in_read ? data.read() : data_dummy;
        ac_array<data_T, AC_BUS_WORDS> din_array;
        //din_array[0] = din;

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
                        in_vector_t in_elem = single_window[i][j];
                        #pragma hls_unroll yes
                        ChannelIndex: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                            kernel_data[i * CONFIG_T::filt_width * CONFIG_T::n_chan + j * CONFIG_T::n_chan + k] = in_base_t(in_elem[k]);
                        }
                    }
                }
                if (CONFIG_T::strategy == nnet::latency) {
                    dense_latency<in_base_t, out_base_t, typename CONFIG_T::mult_config, nnet::II_RF>(
                        kernel_data, res_out, weights, biases);
                } else {
                    dense_resource<in_base_t, out_base_t, typename CONFIG_T::mult_config>(
                        kernel_data, res_out, weights, biases);
                }
                #pragma hls_unroll yes
                CastLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
                    out_vector[i_ic] = res_out[i_ic];
                }
                res_pack[m]=out_vector;
            }
            res.write(res_pack);
        }
    } while (!eof_out);
}


#pragma hls_design
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
        conv_2d_window_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

#pragma hls_design
template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_cl(
    ac_channel<data_T> &data, ac_channel<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    // To ensure only the required implementation methods compile.
    if constexpr (CONFIG_T::implementation == conv_implementation::linebuffer) {
        // std::cout << "INFO: Selected implementation: linebuffer\n";
        conv_2d_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }

    if constexpr (CONFIG_T::implementation == conv_implementation::encoded) {
        // std::cout << "INFO: Selected implementation: encoded\n";
        conv_2d_encoded_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }

    if constexpr (CONFIG_T::implementation == conv_implementation::ac_window) {
        // std::cout << "INFO: Selected implementation: ac_window\n";
        conv_2d_window_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}

} // namespace nnet
#endif

