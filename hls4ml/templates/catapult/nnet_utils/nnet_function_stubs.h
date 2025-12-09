#ifndef NNET_FUNCTION_STUBS_H_
#define NNET_FUNCTION_STUBS_H_

#include "nnet_helpers.h"

#include "nnet_common.h"
#include "nnet_mult.h"
#include <ac_channel.h>

namespace nnet {

template <class data_T, typename CONFIG_T> class FillConv1DBuffer {
  public:
    static void fill_buffer(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                            data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
                            const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, typename CONFIG_T> class FillConv2DBuffer {
  public:
    static void
    fill_buffer(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
                const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, class res_T, typename CONFIG_T> class DenseKernel {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        // To be implemented in subclasses
    }
};

template <class data_T, class res_T, typename CONFIG_T> class DepthwiseDenseKernel {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        // To be implemented in subclasses
    }
};

template <class data_T, class res_T, typename CONFIG_T> class Conv1DKernel {
  public:
    static void conv(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                     typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                     typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
        // To be implemented in subclasses
    }
};

template <int s, int b, int i, ac_q_mode Q, ac_o_mode O, int N> ac_fixed<b, i + s, true> bit_shift(ac_fixed<b, i, true, Q, O> x) {
    // #pragma HLS INLINE
    ac_fixed<b, i + s, true> r;
    r.range() = x.range();
    return r;
};

template <int s, int b, int i, ac_q_mode Q, ac_o_mode O, int N> ac_fixed<b, i + s, false> bit_shift(ac_fixed<b, i, false, Q, O> x) {
    // #pragma HLS INLINE
    ac_fixed<b, i + s, false> r;
    r.range() = x.range();
    return r;
};

template <int s, int b> ac_fixed<b, s, true> bit_shift(ac_int<b,true> x) {
    // #pragma HLS INLINE
    ac_fixed<b, s, true> r;
    r.range() = x.range();
    return r;
};

template <int s, int b> ac_fixed<b, s, false> bit_shift(ac_int<b,false> x) {
    // #pragma HLS INLINE
    ac_fixed<b, s, false> r;
    r.range() = x.range();
    return r;
};
// hls4ml insert code

} // namespace nnet

#endif
