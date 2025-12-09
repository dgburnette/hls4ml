#ifndef NNET_BATCHNORM_H_
#define NNET_BATCHNORM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include <ac_channel.h>
#include <math.h>

namespace nnet {

struct batchnorm_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const int n_filt = -1;
    static const unsigned n_scale_bias = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void normalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
               typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
               typename CONFIG_T::bias_t bias[CONFIG_T::n_scale_bias]) 
{
    data_T cache;

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor

    int multiplier_limit = ceil(float(CONFIG_T::n_in) / float(CONFIG_T::reuse_factor));
    CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::limit(multiplier_limit);

    // Calculate result
    Result: for (int ires = 0; ires < CONFIG_T::n_in; ires++) {
        if (CONFIG_T::n_filt == -1) {
            res[ires] = CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[ires]) +
                        bias[ires];
        } else {
            int norm_index = ires % CONFIG_T::n_filt;
            res[ires] =
                CONFIG_T::template product<data_T, typename CONFIG_T::scale_t>::product(data[ires], scale[norm_index]) +
                bias[norm_index];
        }
    }
}

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void normalize(data_T data[CONFIG_T::n_in], ac_sync &sync_data,
               res_T res[CONFIG_T::n_in],ac_sync &sync_res,
               typename CONFIG_T::scale_t scale[CONFIG_T::n_scale_bias],
               typename CONFIG_T::bias_t bias[CONFIG_T::n_scale_bias]) 
{
  sync_data.sync_in();
  normalize<data_T, res_T, CONFIG_T>(data, res, scale, bias);
  sync_res.sync_out();
}
// ****************************************************
//       Merged Batch Normalization and Quantized Tanh
// ****************************************************
struct batchnorm_quantized_tanh_config {
    // Layer Sizes
    static const unsigned n_in = 10;
    static const int n_filt = -1;
    static const unsigned n_scale_bias = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
};

template <class data_T, typename CONFIG_T>
void normalize_binary_tanh(data_T data[CONFIG_T::n_in], ac_int<1, false> res[CONFIG_T::n_in],
                           data_T threshold[CONFIG_T::n_in]) 
{
    data_T datareg;
    ac_int<1, false> cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        int norm_index = CONFIG_T::n_filt == -1 ? ii : ii % CONFIG_T::n_filt;
        if (datareg >= threshold[norm_index])
            cache = 1;
        else
            cache = 0;

        res[ii] = cache;
    }
}

#pragma hls_design block
template <class data_T, typename CONFIG_T>
void normalize_binary_tanh(data_T data[CONFIG_T::n_in], ac_sync &sync_data,
                           ac_int<1, false> res[CONFIG_T::n_in], ac_sync &sync_res,
                           data_T threshold[CONFIG_T::n_in]) 
{
  sync_data.sync_in();
  normalize_binary_tanh<data_T, CONFIG_T>(data, res, threshold);
  sync_res.sync_out();
}
  
template <class data_T, typename CONFIG_T>
void normalize_ternary_tanh(data_T data[CONFIG_T::n_in], ac_int<2, true> res[CONFIG_T::n_in],
                            data_T threshold_hi[CONFIG_T::n_in], data_T threshold_lo[CONFIG_T::n_in]) 
{
    data_T datareg;
    ac_int<2, true> cache;
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        datareg = data[ii];
        int norm_index = CONFIG_T::n_filt == -1 ? ii : ii % CONFIG_T::n_filt;
        if (datareg > threshold_hi[norm_index])
            cache = 1;
        else if (datareg <= threshold_lo[norm_index])
            cache = -1;
        else
            cache = 0;

        res[ii] = cache;
    }
}

#pragma hls_design block
template <class data_T, typename CONFIG_T>
void normalize_ternary_tanh(data_T data[CONFIG_T::n_in], ac_sync &sync_data,
                            ac_int<2, true> res[CONFIG_T::n_in], ac_sync &sync_res,
                            data_T threshold_hi[CONFIG_T::n_in], data_T threshold_lo[CONFIG_T::n_in]) 
{
  sync_data.sync_in();
  normalize_ternary_tanh<data_T, CONFIG_T>(data, res, threshold_hi, threshold_lo);
  sync_res.sync_out();
}

} // namespace nnet

#endif

