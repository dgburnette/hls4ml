
#ifndef NNET_DENSE_LATENCY_H_
#define NNET_DENSE_LATENCY_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include "nnet_bias.h"
#include <ac_channel.h>
#include <math.h>

namespace nnet {

enum DenseImplType {
    UF_RF,  // Use first dense_latency version
    II_RF   // Use second dense_latency version
};

// ---- detection: does weight_t have .sign and .weight? ----
template <typename W, typename = void>
struct has_sign_weight : std::false_type {};

template <typename W>
struct has_sign_weight<W, decltype(
    (void) std::declval<W>().sign,
    (void) std::declval<W>().weight,
    void()
)> : std::true_type {};

// internal impl: PO2  ----
template <class data_T, class res_T, typename CONFIG_T>
static inline void dense_latency_impl_pow2(
    data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out])
{
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

    (void)ce_reuse_factor; (void)prod1_unroll; (void)prod2_unroll;

    using acc_t = typename CONFIG_T::accum_t;

    acc_t acc[CONFIG_T::n_out];

    #pragma hls_unroll
    ResetAccum: for (unsigned int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = nnet::bias_to_accum<acc_t>(biases[iacc]);
    }

    #pragma hls_unroll prod1_unroll
    Compute: for (unsigned int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T cache = data[ii];

        #pragma hls_unroll prod2_unroll
        MultAndAccum: for (unsigned int jj = 0; jj < CONFIG_T::n_out; jj++) {
            const auto &w = weights[ii * CONFIG_T::n_out + jj];

            acc_t m = static_cast<acc_t>(cache);
            int e = w.weight.to_int();
            if (e >= 0) m <<=  e;
            else        m >>= -e;

            const bool neg = (w.sign != 1); 

            // Work on two's-complement mthd
            const int W = acc_t::width;
            ac_int<W, true> m_bits = m.template slc<W>(0);
            ac_int<W, true> mask   = neg ? ac_int<W, true>(-1) : ac_int<W, true>(0);

            // Two's complement method
            ac_int<W, true> addend_bits = (m_bits ^ mask);
            addend_bits = addend_bits + ac_int<W, true>(neg);
            acc_t addend; addend.set_slc(0, addend_bits);

            acc[jj] = acc[jj] + addend;
        }
    }

    #pragma hls_unroll
    Result: for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}


// ----generic multiply ----
template <class data_T, class res_T, typename CONFIG_T>
void dense_latency_impl_generic(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) 
{
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    // std::cout << "=== dense_latency: UF_RF version selected ===" << std::endl;

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
    //typename CONFIG_T::accum_t mult[CONFIG_T::n_in * CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Initialize accumulator with input biases
    #pragma hls_unroll
    ResetAccum: for (unsigned int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = nnet::bias_to_accum<typename CONFIG_T::accum_t>(biases[iacc]);
    }

    // Perform matrix multiplication and accumulate                  results directly
    #pragma hls_unroll prod1_unroll
    Compute: for (unsigned int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T cache = data[ii];
        #pragma hls_unroll prod2_unroll
        MultAndAccum: for (unsigned int jj = 0; jj < CONFIG_T::n_out; jj++) {
            // Direct accumulation into acc array
            acc[jj] += CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(cache, weights[ii * CONFIG_T::n_out + jj]);
            // acc[jj] += data[ii] * weights[ii * CONFIG_T::n_out + jj];
        }
    }

    // Cast to "res_t" type
    #pragma hls_unroll
    Result: for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++) {
        // res[ires] = (res_T) (acc[ires]);
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}


// ---- single public entry: compile-time dispatch ----
template <class data_T, class res_T, typename CONFIG_T,
          DenseImplType IMPL = UF_RF,
          typename std::enable_if<IMPL == UF_RF, int>::type = 0>
void dense_latency(
    data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_out])
{
    using W = typename CONFIG_T::weight_t;
    if constexpr (has_sign_weight<W>::value) {
        dense_latency_impl_pow2<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        dense_latency_impl_generic<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
}



template <class data_T, class res_T, typename CONFIG_T, DenseImplType IMPL, typename std::enable_if<IMPL == II_RF, int>::type = 0>
void dense_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                   typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                   typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    // std::cout << "=== dense_latency: II_RF version selected ===" << std::endl;

    // Perform matrix multiplication, accumulate results, and cast to "res_t" type in a single loop
    #pragma hls_unroll
    for (unsigned int i = 0; i < CONFIG_T::n_out; i++) {
        acc[i] = (typename CONFIG_T::accum_t)biases[i]; // Initialize accumulator with input biases
        #pragma hls_unroll
        for (unsigned int j = 0; j < CONFIG_T::n_in; j++) {
            acc[i] += data[j] * weights[j * CONFIG_T::n_out + i]; // Accumulate results
        }
        res[i] = cast<data_T, res_T, CONFIG_T>(acc[i]); // Cast to "res_t" type
    }
}


} // namespace nnet

#endif
