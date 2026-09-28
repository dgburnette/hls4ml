#ifndef NNET_LAYERNORM_H_
#define NNET_LAYERNORM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include <ac_fixed.h>
#include <ac_math/ac_inverse_sqrt_pwl.h>
#include <ac_sync.h>

namespace nnet {

struct layernorm_config {
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;
    typedef float accum_t;
    typedef float table_t;

    // Layer Sizes
    static const unsigned n_in = 20;
    static const unsigned seq_len = 4;
    static const unsigned axis = 2;
    static const unsigned epsilon_power_of_10 = 3;
    static const unsigned table_range_power2 = 0;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

// 10^-p as a compile-time constant, without calling pow(): Catapult's synthesis 'analyze' stage has
// no synthesizable definition for pow() (CIN-16 "No definition for routine 'pow'"), even for an
// exponent that's a template/compile-time constant.
constexpr float negative_pow10(unsigned p) { return (p == 0) ? 1.0f : 0.1f * negative_pow10(p - 1); }

template <class data_T, class res_T, typename CONFIG_T>
void layernorm_1d(data_T data[CONFIG_T::n_in / CONFIG_T::seq_len], res_T res[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                  typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor

    typename CONFIG_T::table_t deno_inver = 0;

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    typename CONFIG_T::accum_t sum_cache = 0;
    typename CONFIG_T::accum_t sum_cache2 = 0;
    typename CONFIG_T::accum_t var, mean, diff;
    typename CONFIG_T::accum_t data_diff[dim];

    const typename CONFIG_T::accum_t k_inv = 1.0 / dim;

LAYERNORM_1D_SUM:
    for (int i = 0; i < dim; ++i) {
        sum_cache += static_cast<typename CONFIG_T::accum_t>(data[i]);
    }
    mean = CONFIG_T::template product<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t>::product(sum_cache, k_inv);

LAYERNORM_1D_VAR:
    for (int i = 0; i < dim; ++i) {
        data_diff[i] = static_cast<typename CONFIG_T::accum_t>(data[i]) - mean;
        diff = data_diff[i] * data_diff[i];
        sum_cache2 += diff;
    }
    var = CONFIG_T::template product<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t>::product(sum_cache2, k_inv);

    // 1/sqrt(var + epsilon) via ac_math's synthesizable PWL inverse-sqrt -- the same primitive
    // nnet_multiheadattention_stream.h already uses for its scaling factor -- instead of a
    // hand-rolled lookup table. ac_inverse_sqrt_pwl normalizes the input internally, so (unlike the
    // old table_range_power2-bounded table) it handles any positive var magnitude correctly.
    constexpr float epsilon = negative_pow10(CONFIG_T::epsilon_power_of_10);
    ac_fixed<32, 16, false> inv_sqrt_in = var + (typename CONFIG_T::accum_t)epsilon;
    ac_fixed<32, 16, false> inv_sqrt_out;
    ac_math::ac_inverse_sqrt_pwl(inv_sqrt_in, inv_sqrt_out);
    deno_inver = inv_sqrt_out;

LAYERNORM_1D_RESULT:
    for (int i = 0; i < dim; ++i) {
        res[i] = data_diff[i] * deno_inver * scale[i] + bias[i];
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in],
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    data_T in_val[dim];
    res_T outval[dim];

LAYERNORM_SEQ_LOOP:
    for (int j = 0; j < CONFIG_T::seq_len; ++j) {
        constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
        (void)ce_reuse_factor;
        #pragma hls_pipeline_init_interval ce_reuse_factor
    LAYERNORM_LOAD:
        for (int i = 0; i < dim; ++i) {
            #pragma hls_unroll
            in_val[i] = data[j * dim + i];
        }
        layernorm_1d<data_T, res_T, CONFIG_T>(in_val, outval, scale, bias);
    LAYERNORM_STORE:
        for (int i = 0; i < dim; ++i) {
            #pragma hls_unroll
            res[j * dim + i] = outval[i];
        }
    }
}

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(data_T data[CONFIG_T::n_in], ac_sync &sync_data,
                    res_T res[CONFIG_T::n_in], ac_sync &sync_res,
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len]) {
    sync_data.sync_in();
    layernormalize<data_T, res_T, CONFIG_T>(data, res, scale, bias);
    sync_res.sync_out();
}

} // namespace nnet

#endif
