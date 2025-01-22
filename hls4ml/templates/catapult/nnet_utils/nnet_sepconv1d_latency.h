#ifndef NNET_SEPARABLE_CONV2D_LATENCY_H_
#define NNET_SEPARABLE_CONV2D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include <cstdlib>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_latency_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                                  res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                                  const typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_filt],
                                  const typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {

    constexpr unsigned mult_n_in = CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_acc = CONFIG_T::filt_width;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];

    typename CONFIG_T::accum_t mult[mult_n_in];

    typename CONFIG_T::accum_t acc[mult_n_out];


    // Limit multipliers to control parallelization

#pragma hls_pipeline_init_interval ce_reuse_factor
PartitionLoop:
    for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    #pragma hls_unroll
    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {

            data_T cache;

        // Do the matrix-multiply
        #pragma hls_unroll
        Product:
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
                cache = data_buf[i_pxl][i_in];
                mult[i_in] =
                    CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                        cache, weights[i_in]);
            }

        // Initialize accumulator with input biases
        #pragma hls_unroll
        ResetAccum:
            for (int i_acc = 0; i_acc < mult_n_out; i_acc++) {
                acc[i_acc] = (typename CONFIG_T::accum_t)biases[i_acc];
            }

        // Accumulate multiplication result
        #pragma hls_unroll
        Accum1:
            for (int i_in = 0; i_in < mult_n_acc; i_in++) {
            #pragma hls_unroll
            Accum2:
                for (int i_out = 0; i_out < mult_n_out; i_out++) {
                    acc[i_out] += mult[i_in * mult_n_out + i_out];
                }
            }

        // Cast to "res_t" type
        #pragma hls_unroll
        Result:
            for (int i_res = 0; i_res < mult_n_out; i_res++) {
                *(res++) = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_res]);
            }
        }
    }
}

} // namespace nnet
#endif
