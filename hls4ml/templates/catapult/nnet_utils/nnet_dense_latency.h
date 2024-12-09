
#ifndef NNET_DENSE_LATENCY_H_
#define NNET_DENSE_LATENCY_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <ac_channel.h>
#include <math.h>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void dense_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;

    constexpr int prod1_unroll =
        ((CONFIG_T::n_in % ce_reuse_factor) == 0)
            ? (ce_reuse_factor >= CONFIG_T::n_in ? 1 : (int)(CONFIG_T::n_in / ce_reuse_factor))
        : ((CONFIG_T::n_out % ce_reuse_factor) == 0)
            ? (ce_reuse_factor <= CONFIG_T::n_in
                   ? CONFIG_T::n_in
                   : (int)((CONFIG_T::n_in * CONFIG_T::n_out) + ce_reuse_factor - 1) / ce_reuse_factor)
        : (CONFIG_T::n_out < CONFIG_T::n_in)
            ? (ce_reuse_factor >= CONFIG_T::n_in ? 1 : (int)((CONFIG_T::n_in) + ce_reuse_factor - 1) / ce_reuse_factor)
        : (CONFIG_T::n_in < CONFIG_T::n_out)
            ? (ce_reuse_factor <= CONFIG_T::n_in
                   ? CONFIG_T::n_in
                   : (int)((CONFIG_T::n_in * CONFIG_T::n_out) + ce_reuse_factor - 1) / ce_reuse_factor)
            : 0;

    constexpr int prod2_unroll =
        (CONFIG_T::n_in % ce_reuse_factor == 0)
            ? (ce_reuse_factor <= CONFIG_T::n_in
                   ? CONFIG_T::n_out
                   : (int)((CONFIG_T::n_in * CONFIG_T::n_out) + ce_reuse_factor - 1) / ce_reuse_factor)
        : (CONFIG_T::n_out % ce_reuse_factor == 0) ? (int)((CONFIG_T::n_out) + ce_reuse_factor - 1) / ce_reuse_factor
        : (CONFIG_T::n_out < CONFIG_T::n_in)
            ? (ce_reuse_factor <= CONFIG_T::n_in
                   ? CONFIG_T::n_out
                   : (int)((CONFIG_T::n_in * CONFIG_T::n_out) + ce_reuse_factor - 1) / ce_reuse_factor)
        : (CONFIG_T::n_in < CONFIG_T::n_out) ? (int)((CONFIG_T::n_out) + ce_reuse_factor - 1) / ce_reuse_factor
                                             : 0;

    (void)ce_reuse_factor; // to silence compiler warnings
    (void)prod2_unroll;
    (void)prod1_unroll;

    data_T cache;
    // coverity[STACK_USE]
    // typename CONFIG_T::accum_t mult[CONFIG_T::n_in * CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

// Initialize accumulator with input biases
#pragma hls_unroll
ResetAccum:
    for (unsigned int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

// Perform matrix multiplication and accumulate results directly
#pragma hls_unroll prod1_unroll
Compute:
    for (unsigned int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS PIPELINE
        // data_T cache = data[ii];
        #pragma hls_unroll prod2_unroll
    MultAndAccum:
        for (unsigned int jj = 0; jj < CONFIG_T::n_out; jj++) {
            // Direct accumulation into acc array
            // acc[jj] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(cache, weights[index]);
            acc[jj] += data[ii] * weights[ii * CONFIG_T::n_out + jj];
        }
    }

// Cast to "res_t" type
#pragma hls_unroll
Result:
    for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        // res[ires] = (res_T) (acc[ires]);
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}
} // namespace nnet

#endif
