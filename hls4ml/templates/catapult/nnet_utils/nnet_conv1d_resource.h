#ifndef NNET_CONV1D_RESOURCE_H_
#define NNET_CONV1D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template <class data_T, typename CONFIG_T>
void im2col_1d(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
               data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::out_width]) {
    // int index = 0;
    for (int channel = CONFIG_T::n_chan; channel--; data += CONFIG_T::in_width) {
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            #pragma hls_unroll
            int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation;
            for (int output_col = CONFIG_T::out_width; output_col; output_col--) {
                #pragma hls_unroll
                if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                    *(data_col++) = data[input_col];
                    // data_col[index] = data[input_col];
                } else {
                    *(data_col++) = 0;
                    // data_col[index] = 0;
                }
                // index++;
                input_col += CONFIG_T::stride_width;
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_full(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                  typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                  typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    data_T data_conv[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::out_width];
    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];


    im2col_1d<data_T, CONFIG_T>(data, data_conv);

    for (int i = 0; i < CONFIG_T::out_width; i++) {
        #pragma hls_unroll
        for (int j = 0; j < CONFIG_T::filt_width * CONFIG_T::n_chan; j++) {
            data_col[j] = data_conv[j * CONFIG_T::out_width + i];
        }
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            // res[i * CONFIG_T::n_filt + j] = res_col[j];
            res[j * CONFIG_T::out_width + i] = res_col[j]; // Transposed order
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_1d_cf_idx(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                      data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan], const int col) {
ChannelLoop:
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
        //#pragma hls_unroll
    KernelLoop:
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            #pragma hls_unroll
            int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation + col * CONFIG_T::stride_width;
            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                //*(data_col++) = data[input_col];
                data_col[channel * CONFIG_T::filt_width + kernel_col] = data[channel * CONFIG_T::in_width + input_col];
            } else {
                //*(data_col++) = 0;
                data_col[channel * CONFIG_T::filt_width + kernel_col] = 0;
            }
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_1d_cf(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                  data_T data_col[CONFIG_T::n_chan * CONFIG_T::filt_width], const int col) {
    int index = 0;
ChannelLoop:
    for (int channel = CONFIG_T::n_chan; channel--; data += CONFIG_T::in_width) {
        #pragma hls_unroll
    KernelLoop:
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation + col * CONFIG_T::stride_width;
            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                //*(data_col++) = data[input_col];
                data_col[index] = data[input_col];
            } else {
                //*(data_col++) = 0;
                data_col[index] = 0;
            }
            index++;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_resource_cf(data_T data[CONFIG_T::n_chan * CONFIG_T::in_width],
                         res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                         typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                         typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin * nout, rufactor);

    /// correctly

    data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];


ColLoop:
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        im2col_1d_cf<data_T, CONFIG_T>(data, data_col, i);
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            // res[i * CONFIG_T::n_filt + j] = res_col[j];
            res[j * CONFIG_T::out_width + i] = res_col[j]; // Transposed order
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                  data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan], const int col) {
    int index = 0;
KernelLoop:
    for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
        #pragma hls_unroll

    ChannelLoop:
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            int index_data = (col * CONFIG_T::stride_width + kernel_col - CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;

            if (index_data >= 0 && index_data < CONFIG_T::in_width * CONFIG_T::n_chan) {
                data_col[index] = data[index_data];
            } else {
                data_col[index] = 0;
            }
            index++;
        }
    }
}

template <class data_T, typename CONFIG_T>
void im2col_1d_pointwise_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], data_T data_col[CONFIG_T::n_chan],
                            const int col) {
    int index = 0;
ChannelLoop:
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
        #pragma hls_unroll

        int index_data = (col * CONFIG_T::stride_width - CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;

        if (index_data >= 0 && index_data < CONFIG_T::in_width * CONFIG_T::n_chan) {
            data_col[index] = data[index_data];
        } else {
            data_col[index] = 0;
        }
        index++;
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_resource_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                         res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                         typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                         typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned block_factor = DIV_ROUNDUP(mult_n_in * mult_n_out, CONFIG_T::reuse_factor);
    constexpr unsigned multscale = block_factor / mult_n_out;

    assert((block_factor % mult_n_out == 0 || CONFIG_T::reuse_factor >= mult_n_in) &&
           "The current Reuse Factor is not allowed");
    assert((CONFIG_T::reuse_factor <= CONFIG_T::filt_width * CONFIG_T::n_chan) &&
           "This function is correct only for RF <= FILT_WIDTH * N_CHAN");

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][mult_n_out];

//#pragma hls_unroll // We don't want this loop unrolled
PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    #pragma hls_unroll
    PixelInitAccumLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {

        #pragma hls_unroll
        InitAccumLoop:
            for (unsigned i_acc = 0; i_acc < mult_n_out; i_acc++) {
                acc[i_pxl][i_acc] = (typename CONFIG_T::accum_t)biases[i_acc];
            }
        }

    #pragma hls_pipeline_init_interval 1
    ReuseLoop:
        for (unsigned i_rf = 0; i_rf < CONFIG_T::reuse_factor; i_rf++) {
            unsigned i_w = i_rf;
            unsigned i_in = i_rf;
            unsigned i_out = 0;
            unsigned i_acc = 0;

        #pragma hls_unroll
        MultLoop:
            for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
            #pragma hls_unroll
            PixelMultLoop:
                for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
                    acc[i_pxl][i_out] += static_cast<typename CONFIG_T::accum_t>(
                        CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                            data_buf[i_pxl][i_in], weights[i_w]));
                }

                // Increment i_w
                i_w += CONFIG_T::reuse_factor;
                // Increment i_in
                i_in += CONFIG_T::reuse_factor;
                if (i_in >= mult_n_in) {
                    i_in = i_rf;
                }
                // Increment i_out
                if (i_acc + 1 >= multscale) {
                    i_acc = 0;
                    i_out++;
                } else {
                    i_acc++;
                }
            }
        }

    #pragma hls_unroll
    PixelResultLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
        // Cast to "res_t" type
        #pragma hls_unroll
        ResultLoop:
            for (unsigned i_res = 0; i_res < mult_n_out; i_res++) {
                *(res++) = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_pxl][i_res]);
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_resource_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                                   res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                                   typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                                   typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    assert(CONFIG_T::filt_width == 1);

    const int nin = CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin * nout, rufactor);

    /// correctly

    data_T data_col[CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];


ColLoop:
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        im2col_1d_pointwise_cl<data_T, CONFIG_T>(data, data_col, i);
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            res[i * CONFIG_T::n_filt + j] = res_col[j];
        }
    }
}

} // namespace nnet
#endif
