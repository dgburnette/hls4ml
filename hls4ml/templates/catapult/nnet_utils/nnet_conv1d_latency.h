#ifndef NNET_CONV1D_LATENCY_H_
#define NNET_CONV1D_LATENCY_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

// Computes multiplier limit
// This function should not be synthesized into firmware
template <typename CONFIG_T>
int compute_multiplier_limit(
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt]) {
    int n_mult = 0;
    for (int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            for (int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                for (int jj = 0; jj < CONFIG_T::filt_width; jj++) {

                    int index_weight = jj * CONFIG_T::n_chan * CONFIG_T::n_filt + cc * CONFIG_T::n_filt + ff;

                    if ((ii * CONFIG_T::stride_width + jj) < CONFIG_T::pad_left ||
                        (ii * CONFIG_T::stride_width + jj) >= (CONFIG_T::pad_left + CONFIG_T::in_width)) {
                        // padded -- do nothing
                        continue;
                    } else {
                        // need to tune this cut?
                        if (weights[index_weight] > 1e-20 || weights[index_weight] < -1e-20) {
                            n_mult++;
                        } // end if nonzero weight
                    }     // end not padding
                }         // end loop accross filter
            }             // end channel loop
        }                 // end filter loop
    }                     // end output loop

    return DIV_ROUNDUP(n_mult, CONFIG_T::reuse_factor);

} // end compute_n_mult

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_latency_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                        res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                        typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;

    constexpr unsigned mult_n_in = CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    typename CONFIG_T::accum_t mult[mult_n_in * mult_n_out];
    typename CONFIG_T::accum_t acc[mult_n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases

    // Parallel mode

    // Limit multipliers to control parallelization
    const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);

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
        Product1:
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
                cache = data_buf[i_pxl][i_in];
            #pragma hls_unroll
            Product2:
                for (int i_out = 0; i_out < mult_n_out; i_out++) {
                    mult[i_in * mult_n_out + i_out] =
                        CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                            cache, weights[i_in * mult_n_out + i_out]);
                }
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
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
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
                *(res++) = (res_T)acc[i_res]; // cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_res]);
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor],
                                  res_T res[CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor],
                                  typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                                  typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    assert(CONFIG_T::filt_width == 1);

    typename CONFIG_T::accum_t mult[CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan / CONFIG_T::reuse_factor];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_width / CONFIG_T::reuse_factor][CONFIG_T::n_filt];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases

    // Parallel mode

    // Limit multipliers to control parallelization
    const int multiplier_limit = compute_multiplier_limit<CONFIG_T>(weights);

// Convolve, saving all multiplication results to accumulate later
ConvOut:
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
    ConvFilt:
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        #pragma hls_unroll
        ConvChan:
            for (int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult = ii * CONFIG_T::n_filt * CONFIG_T::n_chan + ff * CONFIG_T::n_chan + cc;
                int index_weight = cc * CONFIG_T::n_filt + ff;
                int index_data = (ii * CONFIG_T::stride_width - CONFIG_T::pad_left) * CONFIG_T::n_chan + cc;

                if ((ii * CONFIG_T::stride_width) < CONFIG_T::pad_left ||
                    (ii * CONFIG_T::stride_width) >= (CONFIG_T::pad_left + CONFIG_T::in_width)) {
                    mult[index_mult] = 0;
                } else {
                    mult[index_mult] = CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::weight_t>::product(
                        data[index_data], weights[index_weight]);
                }
            } // end channel loop
        }     // end filter loop
    }         // end output loop

    // Initialize accumulator with input biases
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
        #pragma hls_unroll
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            acc[ii][ff] = biases[ff];
        }
    }

// Accumulate multiplication result
AccumOut:
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
    AccumFilt:
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Do "dot product" sum within filter and sum over channels
        AccumChan:
            for (int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult = ii * CONFIG_T::n_filt * CONFIG_T::n_chan + ff * CONFIG_T::n_chan + cc;
                acc[ii][ff] += mult[index_mult];
            } // end channel loop
        }     // end filter loop
    }         // end output loop

    // Cast to "res_t" type
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
        #pragma hls_unroll
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            res[ii * CONFIG_T::n_filt + ff] = (res_T)acc[ii][ff]; // cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[ii][ff]);
        }
    }
}

} // namespace nnet
#endif
