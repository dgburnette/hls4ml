#ifndef NNET_CONV2D_WINDOW_V2_MIXED_CLASSES_H_
#define NNET_CONV2D_WINDOW_V2_MIXED_CLASSES_H_

#include <assert.h>

#include <ac_array.h>
#include <ac_channel.h>
#include <ac_int.h>

#include <ac_ipl/ac_window_v2.h>
#include <ac_ipl/ac_window_v2_flush.h>

#include "nnet_utils/nnet_common.h"
#include "nnet_utils/nnet_helpers.h"

namespace nnet {

// ============================================================
// SAME padding (AC_CONSTANT) using ac_window_v2_flush_2d (class)
// ============================================================
template <class data_T, class res_T, typename CONFIG_T>
class conv2d_window_v2_flush_same_cl {
public:
  // Explicit default constructor to ensure window is initialized
  conv2d_window_v2_flush_same_cl() : window() {}
  static constexpr int IMG_H = CONFIG_T::in_height;
  static constexpr int IMG_W = CONFIG_T::in_width;
  static constexpr int WIN_H = CONFIG_T::filt_height;
  static constexpr int WIN_W = CONFIG_T::filt_width;

  typedef typename data_T::ElemType        in_vector_t;
  typedef typename res_T::ElemType         out_vector_t;
  typedef typename in_vector_t::base_type  in_base_t;
  typedef typename out_vector_t::base_type out_base_t;

  static constexpr ac_buff_arch_flush BUFF_TYPE =
      (IMG_W % 2 == 0) ? AC_SPWRMASK_FLUSH : AC_1R1W_FLUSH;
  static constexpr ac_padding_method AC_PMODE = AC_CONSTANT;

  typedef ac_window_v2_flush_2d<
      in_vector_t,
      IMG_H, IMG_W,
      WIN_H, WIN_W,
      BUFF_TYPE, AC_PMODE,
      AC_BUS_WORDS>
      WINDOW_2D_TYPE;

  #pragma hls_design interface
  #pragma hls_pipeline_init_interval 1
  void run(ac_channel<data_T> &data,
           ac_channel<res_T>  &res,
           typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width *
                                              CONFIG_T::n_chan * CONFIG_T::n_filt],
           typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    // SAME padding (AC_CONSTANT) path
    assert((AC_BUS_WORDS <= IMG_W) &&
           "ac_window does not support AC_BUS_WORDS > feature width.");

    static constexpr int ce_reuse_factor =
        (CONFIG_T::strategy == nnet::latency) ? CONFIG_T::reuse_factor : 1;

    const ac_int<ac::nbits<IMG_W>::val,  false> width  = IMG_W;
    const ac_int<ac::nbits<IMG_H>::val, false> height = IMG_H;

    constexpr int AC_WORDS = WINDOW_2D_TYPE::AC_WORDS;
    ac_array<in_vector_t, CONFIG_T::filt_height, AC_WORDS> window_out;

    bool sof_out = false, sol_out = false, eol_out = false, vld_out = false;
    bool eof_out = false;
    bool dont_read_data = false;

    data_T din;
    res_T  res_pack;
    out_vector_t out_vec;

    constexpr int PACKED_WIDTH __attribute__((unused)) = WINDOW_2D_TYPE::PACKED_WIDTH;
    constexpr int MAX_ITERS __attribute__((unused)) =
        IMG_H * PACKED_WIDTH + WINDOW_2D_TYPE::NUM_FLUSH_ITERS;

    #pragma hls_iterations MAX_ITERS
    #pragma hls_pipeline_init_interval 1
    do {
      din = 0;
      const bool write = data.nb_read(din);

      window.run(din, write, width, height,
                 window_out, sof_out, eof_out,
                 sol_out, eol_out, vld_out, dont_read_data);

      if (vld_out) {
        res_pack = 0;

        #pragma hls_unroll yes
        for (int m = 0; m < AC_BUS_WORDS; m++) {

          #pragma hls_unroll
          for (unsigned ff = 0; ff < CONFIG_T::n_filt; ff++) {
            typename CONFIG_T::accum_t acc = (typename CONFIG_T::accum_t)biases[ff];

            #pragma hls_unroll
            for (unsigned i = 0; i < CONFIG_T::filt_height; i++) {
              #pragma hls_unroll
              for (unsigned j = 0; j < CONFIG_T::filt_width; j++) {
                const in_vector_t pix = window_out[i][j + m];

                #pragma hls_unroll
                for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                  const unsigned jj =
                      i * CONFIG_T::filt_width * CONFIG_T::n_chan +
                      j * CONFIG_T::n_chan + k;
                  #pragma hls_waive OVL
                  acc += in_base_t(pix[k]) * weights[jj * CONFIG_T::n_filt + ff];
                }
              }
            }

            out_vec[ff] = cast<in_base_t, out_base_t, CONFIG_T>(acc);
          }

          res_pack[m] = out_vec;
        }

        res.write(res_pack);
      }

    } while (!eof_out);
  }

private:
  WINDOW_2D_TYPE window; // keep member for flush framing
};

// ============================================================
// VALID padding (AC_NO_PADDING) using ac_window_v2_2d (valid only)
// ============================================================
template <class data_T, class res_T, typename CONFIG_T>
class conv2d_window_v2_valid_cl {
public:
  // Explicit default constructor to ensure window is initialized
  conv2d_window_v2_valid_cl() : window() {}
  static constexpr int IMG_H = CONFIG_T::in_height;
  static constexpr int IMG_W = CONFIG_T::in_width;
  static constexpr int WIN_H = CONFIG_T::filt_height;
  static constexpr int WIN_W = CONFIG_T::filt_width;

  typedef typename data_T::ElemType        in_vector_t;
  typedef typename res_T::ElemType         out_vector_t;
  typedef typename in_vector_t::base_type  in_base_t;
  typedef typename out_vector_t::base_type out_base_t;

  static constexpr ac_buff_type BUFF_TYPE =
      (IMG_W % 2 == 0) ? AC_SPWRMASK : AC_DUAL;
  static constexpr ac_padding_method AC_PMODE = AC_NO_PADDING;

  typedef ac_window_v2_2d<
      in_vector_t,
      IMG_H, IMG_W,
      WIN_H, WIN_W,
      BUFF_TYPE, AC_PMODE,
      AC_BUS_WORDS>
      WINDOW_2D_TYPE;

  // #pragma hls_design interface
  // #pragma hls_pipeline_init_interval 1
  void run(ac_channel<data_T> &data,
           ac_channel<res_T>  &res,
           typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width *
                                              CONFIG_T::n_chan * CONFIG_T::n_filt],
           typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    // VALID padding path only
    assert((AC_BUS_WORDS <= IMG_W) &&
           "ac_window valid does not support AC_BUS_WORDS > feature width.");
    assert(!(AC_BUS_WORDS > WIN_W) &&
           "ac_window with 'valid' padding does not support AC_BUS_WORDS > WIN_W.");
    assert(!(((WIN_W - 1) % AC_BUS_WORDS) != 0) &&
           "ac_window with 'valid' padding requires (WIN_W - 1) divisible by AC_BUS_WORDS.");

    static constexpr int ce_reuse_factor =
        (CONFIG_T::strategy == nnet::latency) ? CONFIG_T::reuse_factor : 1;

    const ac_int<ac::nbits<IMG_W>::val,  false> width  = IMG_W;
    const ac_int<ac::nbits<IMG_H>::val, false> height = IMG_H;

    bool eof_out = false;
    bool in_read = true;

    res_T res_pack;
    out_vector_t out_vec;

    // valid: no extra flush iters
    constexpr int ITERATIONS __attribute__((unused)) = IMG_W * IMG_H;
    #pragma hls_iterations ITERATIONS
    #pragma hls_pipeline_init_interval 1
    do {
      data_T din = data.read();

      constexpr int AC_WORDS = WINDOW_2D_TYPE::AC_WORDS;
      ac_array<in_vector_t, CONFIG_T::filt_height, AC_WORDS> window_out;

      bool sof_out = false, sol_out = false, eol_out = false;
      ac_int<AC_BUS_WORDS, false> vld_mask = 0;

      window.run(din, width, height, in_read,
                 window_out, sof_out, eof_out, sol_out, eol_out, vld_mask);

      if (vld_mask != 0) {
        res_pack = 0;

        #pragma hls_unroll yes
        for (int m = 0; m < AC_BUS_WORDS; m++) {
          // If library provides per-lane validity, respect it.
          if (!vld_mask[m]) continue;

          #pragma hls_unroll
          for (unsigned ff = 0; ff < CONFIG_T::n_filt; ff++) {
            typename CONFIG_T::accum_t acc = (typename CONFIG_T::accum_t)biases[ff];

            #pragma hls_unroll
            for (unsigned i = 0; i < CONFIG_T::filt_height; i++) {
              #pragma hls_unroll
              for (unsigned j = 0; j < CONFIG_T::filt_width; j++) {
                const in_vector_t pix = window_out[i][j + m];

                #pragma hls_unroll
                for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                  const unsigned jj =
                      i * CONFIG_T::filt_width * CONFIG_T::n_chan +
                      j * CONFIG_T::n_chan + k;
                  #pragma hls_waive OVL
                  acc += in_base_t(pix[k]) * weights[jj * CONFIG_T::n_filt + ff];
                }
              }
            }

            out_vec[ff] = cast<in_base_t, out_base_t, CONFIG_T>(acc);
          }

          res_pack[m] = out_vec;
        }

        res.write(res_pack);
      }

    } while (!eof_out);
  }

private:
  WINDOW_2D_TYPE window;
};

// ============================================================
// Dispatcher (compile-time) based on CONFIG_T::padding
//   - SAME  -> flush (AC_CONSTANT)
//   - VALID -> non-flush window (AC_NO_PADDING)
// ============================================================
template <class data_T, class res_T, typename CONFIG_T, padding_type PADDING_T>
struct conv2d_window_v2_dispatch;

template <class data_T, class res_T, typename CONFIG_T>
struct conv2d_window_v2_dispatch<data_T, res_T, CONFIG_T, padding_type::same> {
  static void run(ac_channel<data_T> &data,
                  ac_channel<res_T>  &res,
                  typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width *
                                                     CONFIG_T::n_chan * CONFIG_T::n_filt],
                  typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    static conv2d_window_v2_flush_same_cl<data_T, res_T, CONFIG_T> impl;
    impl.run(data, res, weights, biases);
  }
};

template <class data_T, class res_T, typename CONFIG_T>
struct conv2d_window_v2_dispatch<data_T, res_T, CONFIG_T, padding_type::valid> {
  static void run(ac_channel<data_T> &data,
                  ac_channel<res_T>  &res,
                  typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width *
                                                     CONFIG_T::n_chan * CONFIG_T::n_filt],
                  typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    static conv2d_window_v2_valid_cl<data_T, res_T, CONFIG_T> impl;
    impl.run(data, res, weights, biases);
  }
};

} // namespace nnet

#endif // NNET_CONV2D_WINDOW_V2_MIXED_CLASSES_H_
