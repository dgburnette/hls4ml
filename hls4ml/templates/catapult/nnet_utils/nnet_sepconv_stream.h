#ifndef NNET_SEPARABLE_CONV_STREAM_H_
#define NNET_SEPARABLE_CONV_STREAM_H_

#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include <ac_assert.h>
#include <ac_channel.h>

namespace nnet {

#pragma hls_design inline
template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product(data_T data[CONFIG_T::kernel_size * CONFIG_T::n_chan], res_T res[CONFIG_T::n_filt],
                       typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
                       typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {

    typename CONFIG_T::accum_t mult[CONFIG_T::kernel_size * CONFIG_T::n_filt];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_filt];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    #pragma hls_unroll

    // Add dummy loop to which the pipeline pragma can be applied
    do {

    // Do the matrix-multiply
    #pragma hls_unroll
    Product1:
        for (unsigned int ii = 0; ii < CONFIG_T::kernel_size * CONFIG_T::n_chan; ii++) {
        #pragma hls_unroll
        Product2:
            for (unsigned int jj = 0; jj < CONFIG_T::d_mult; jj++) {
                int index = ii * CONFIG_T::d_mult + jj;
                mult[index] =
                    CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                        data[ii], weights[index]);
            }
        }

    // Initialize accumulator with input biases
    #pragma hls_unroll
    ResetAccum:
        for (unsigned int iacc = 0; iacc < CONFIG_T::n_filt; iacc++) {
            acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
        }

    // Accumulate multiplication result
    #pragma hls_unroll
    Accum1:
        for (unsigned int ii = 0; ii < CONFIG_T::kernel_size; ii++) {
        #pragma hls_unroll
        Accum2:
            for (unsigned int jj = 0; jj < CONFIG_T::n_chan; jj++) {
            #pragma hls_unroll
            Accum3:
                for (unsigned int kk = 0; kk < CONFIG_T::d_mult; kk++) {
                    int index1 = ii * CONFIG_T::n_chan * CONFIG_T::d_mult + jj * CONFIG_T::d_mult + kk;
                    int index2 = jj * CONFIG_T::d_mult + kk;
                    acc[index2] += mult[index1];
                }
            }
        }

    // Cast to "res_t" type
    #pragma hls_unroll
    Result:
        for (unsigned int ires = 0; ires < CONFIG_T::n_filt; ires++) {
            res[ires] = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[ires]);
        }
    } while (0);
}

#pragma hls_design inline
template <class data_T, class res_T, typename CONFIG_T>
void depthwise_mult_buffer(ac_channel<typename data_T::value_type> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
                           res_T &res_pack, ac_channel<res_T> &res_stream, unsigned &outputs_ready,
                           typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
                           typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {

    typename data_T::value_type data[CONFIG_T::kernel_size * CONFIG_T::n_chan];
    typename res_T::value_type res[CONFIG_T::n_chan];

    #pragma hls_unroll
InitData:
    for (unsigned int id = 0; id < CONFIG_T::kernel_size * CONFIG_T::n_chan; id++) {
        data[id] = data_window[id].read();
    }

    if (CONFIG_T::strategy == nnet::latency) {
        depthwise_product<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(data, res, weights, biases);
    } else {
        assert("Resource strategy for DepthwiseConv2D is not supported." && false);
    }

    #pragma hls_unroll
CastLoop:
    for (unsigned jj = 0; jj < CONFIG_T::n_chan; jj++) {
        if (res_T::size / CONFIG_T::n_chan == 1) {
            res_pack[jj] = res[jj];
        } else {
            res_pack[outputs_ready * CONFIG_T::n_chan + jj] = res[jj];
        }
    }

    if (res_T::size / CONFIG_T::n_chan == 1) {
        res_stream.write(res_pack);
    } else {
        if (outputs_ready == (res_T::size / CONFIG_T::n_chan) - 1) {
            res_stream.write(res_pack);
            outputs_ready = 0;
        } else {
            outputs_ready++;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void compute_depthwise_output_encoded(
    const data_T &in_elem, ac_channel<typename data_T::value_type> data_window[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    ac_channel<res_T> &res, res_T &res_pack, unsigned &outputs_ready,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan], ac_int<CONFIG_T::kernel_size, false> *pixel_idx) {

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    #pragma hls_unroll
MultLoop:
    for (unsigned p = 0; p < data_T::size / CONFIG_T::n_chan; p++) {
    #pragma hls_unroll
    CopyDataFilt:
        for (unsigned f = 0; f < CONFIG_T::kernel_size; f++) {
        #pragma hls_unroll
        CopyDataChan:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                if (pixel_idx[p][f])
                    data_window[f * CONFIG_T::n_chan + c].write(in_elem[p * CONFIG_T::n_chan + c]);
            }
        }
        if (pixel_idx[p][CONFIG_T::kernel_size - 1]) {
            depthwise_mult_buffer<data_T, res_T, CONFIG_T>(data_window, res_pack, res, outputs_ready, weights, biases);
        }
    }
}

#pragma hls_design inline
template <class data_T, class res_T, typename CONFIG_T>
void pointwise_mult_buffer(const data_T &data_pack, ac_channel<res_T> &res_stream,
                           typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                           typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {

    typename data_T::value_type data[CONFIG_T::n_chan];

    typename res_T::value_type res[CONFIG_T::n_filt];

    res_T res_pack;
    // PRAGMA_DATA_PACK(res_pack)

    #pragma hls_unroll
InitData:
    for (int id = 0; id < CONFIG_T::n_chan; id++) {
        data[id] = data_pack[id];
    }

    if (CONFIG_T::strategy == nnet::latency) {
        dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(
            data, res, weights, biases);
    } else {
        dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(
            data, res, weights, biases);
    }

    #pragma hls_unroll
CastLoop:
    for (unsigned jj = 0; jj < CONFIG_T::n_filt; jj++) {
        res_pack[jj] = res[jj];
    }

    res_stream.write(res_pack);
}

// Line Buffer Implementation (Phil's)
#pragma hls_design inline
template <class data_T, class res_T, typename CONFIG_T>
void compute_depthwise_output_buffer_1d(const data_T &in_elem, ac_channel<res_T> &res_stream,
                                        typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
                                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {

    // Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;

    // Counters
    static int pX = 0;
    static int sX = 0;

    static typename data_T::value_type kernel_data[CONFIG_T::filt_width * CONFIG_T::n_chan];

    typename res_T::value_type res_out[CONFIG_T::n_filt];

    res_T res_pack;
    // PRAGMA_DATA_PACK(res_pack)

    // Add pixel to buffer
    nnet::kernel_shift_1d<data_T, CONFIG_T>(in_elem, kernel_data);

    // Check to see if we have a full kernel
    if ((sX - lShiftX) == 0 && pX > lShiftX - 1) {
        // Dense multiply
        if (CONFIG_T::strategy == nnet::latency) {
            depthwise_product<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(kernel_data, res_out,
                                                                                                 weights, biases);
        } else {
            assert("Resource strategy for DepthwiseConv1D is not supported." && false);
        }

    // Pack output
    #pragma hls_unroll
    CastLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
            res_pack[i_ic] = res_out[i_ic];
        }

        // Write output to stream when output ready
        res_stream.write(res_pack);
    }

    // Pointer Housekeeping
    if (pX + 1 == CONFIG_T::in_width) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
    } else {
        pX = pX + 1;
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

#pragma hls_design inline
template <class data_T, class res_T, typename CONFIG_T>
void compute_depthwise_output_buffer_2d(const data_T &in_elem,
                                        ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width>
                                            line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan],
                                        ac_channel<res_T> &res_stream,
                                        typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
                                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {

    // Thresholds
    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    // counters
    static int pX = 0; // pixel X
    static int pY = 0; // pixel Y

    static int sX = 0; // stride X
    static int sY = 0; // stride Y

    static typename data_T::value_type kernel_data[CONFIG_T::kernel_size * CONFIG_T::n_chan];

    typename res_T::value_type res_out[CONFIG_T::n_filt];

    res_T res_pack;
    // PRAGMA_DATA_PACK(res_pack)

    // Add pixel to buffer
    nnet::shift_line_buffer<data_T, CONFIG_T>(in_elem, line_buffer, kernel_data);

    // Check to see if we have a full kernel
    if ((sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1) {
        // Dense multiply
        if (CONFIG_T::strategy == nnet::latency) {
            depthwise_product<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(kernel_data, res_out,
                                                                                                 weights, biases);
        } else {
            assert("Resource strategy for DepthwiseConv2D is not supported." && false);
        }

    // Pack output
    #pragma hls_unroll
    CastLoop:
        for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
            res_pack[i_ic] = res_out[i_ic];
        }

        // Write output to stream when output ready
        res_stream.write(res_pack);
    }

    // Pointer Housekeeping
    if (pX + 1 == CONFIG_T::in_width) // Includes padding, end of line (padded)
    {
        pX = 0;
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) { // Reached bottom of image
            pY = 0;
            sY = 0;
        } else {
            pY = pY + 1;
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1;
        }
    } else {
        pX = pX + 1;
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }
}

} // namespace nnet
#endif
