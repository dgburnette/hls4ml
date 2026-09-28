#ifndef NNET_SEP_CONV2D_WINDOW_V2_MIXED_CLASSES_H_
#define NNET_SEP_CONV2D_WINDOW_V2_MIXED_CLASSES_H_

#include <assert.h>

#include <ac_array.h>
#include <ac_channel.h>
#include <ac_int.h>

// NOTE:
// - SAME padding path uses ac_window_v2_flush
// - VALID padding path uses ac_window_v2 (non-flush)

#include <ac_ipl/ac_window_v2.h>
#include <ac_ipl/ac_window_v2_flush.h>

#include "nnet_utils/nnet_common.h"
#include "nnet_utils/nnet_helpers.h"

namespace nnet {

// ============================================================
// 1) SAME padding : AC_WINDOW V2 FLUSH (class-based)
// ============================================================
template <class data_T, class res_T, typename CONFIG_T>
class depthwise_conv2d_window_v2_flush_same_cl {
public:
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

  // #pragma hls_design interface
  // #pragma hls_pipeline_init_interval 1
  void run(ac_channel<data_T> &data,
           ac_channel<res_T> &res,
           typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
           typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    assert((AC_BUS_WORDS <= IMG_W) &&
           "ac_window_flush does not support AC_BUS_WORDS > feature width.");

    static constexpr int d_mult = CONFIG_T::n_filt / CONFIG_T::n_chan;
    static constexpr int ce_reuse_factor =
        CONFIG_T::reuse_factor * (CONFIG_T::strategy == nnet::latency);
    (void)ce_reuse_factor;

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

    typename CONFIG_T::accum_t acc[CONFIG_T::n_filt];
    out_base_t res_out[CONFIG_T::n_filt];

    constexpr int PACKED_WIDTH __attribute__((unused)) = WINDOW_2D_TYPE::PACKED_WIDTH;
    constexpr int MAX_ITERS __attribute__((unused)) =
        IMG_H * PACKED_WIDTH + WINDOW_2D_TYPE::NUM_FLUSH_ITERS;

    #pragma hls_iterations MAX_ITERS
    #pragma hls_pipeline_init_interval 1
    do {
      din = 0;
      bool write = false;

      if (!dont_read_data) {
        write = data.nb_read(din);
      } else {
        write = false;
        din = 0;
      }

      window.run(din, write, width, height,
                 window_out, sof_out, eof_out,
                 sol_out, eol_out, vld_out, dont_read_data);

      if (vld_out) {
        res_pack = 0;

        #pragma hls_unroll yes
        for (int m = 0; m < AC_BUS_WORDS; m++) {

          #pragma hls_unroll
          for (unsigned jj = 0; jj < CONFIG_T::n_chan; jj++) {

            #pragma hls_unroll
            for (unsigned kk = 0; kk < d_mult; kk++) {
              const int filt_idx = jj * d_mult + kk;

              acc[filt_idx] = (typename CONFIG_T::accum_t)biases[filt_idx];

              #pragma hls_unroll
              for (unsigned ii = 0; ii < CONFIG_T::kernel_size; ii++) {
                const unsigned wi = ii / CONFIG_T::filt_width;
                const unsigned wj = ii - wi * CONFIG_T::filt_width;

                const in_base_t pix = in_base_t(window_out[wi][wj + m][jj]);
                const int weight_idx =
                    (ii * CONFIG_T::n_chan + jj) * d_mult + kk;

                acc[filt_idx] += CONFIG_T::mult_config::template product<
                    in_base_t, typename CONFIG_T::mult_config::weight_t>::product(
                    pix, weights[weight_idx]);
              }

              res_out[filt_idx] =
                  cast<in_base_t, out_base_t, typename CONFIG_T::mult_config>(acc[filt_idx]);
            }
          }

          #pragma hls_unroll yes
          for (unsigned ff = 0; ff < CONFIG_T::n_filt; ff++) {
            out_vec[ff] = res_out[ff];
          }

          res_pack[m] = out_vec;
        }

        res.write(res_pack);
      }

    } while (!eof_out);
  }

private:
  WINDOW_2D_TYPE window; // must persist for flush framing
};

// ============================================================
// 2) VALID padding : AC_WINDOW V2 (non-flush, class-based)
// ============================================================
template <class data_T, class res_T, typename CONFIG_T>
class depthwise_conv2d_window_v2_valid_cl {
public:
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

  #pragma hls_design interface
  #pragma hls_pipeline_init_interval 1
  void run(ac_channel<data_T> &data,
           ac_channel<res_T> &res,
           typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
           typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {
    // VALID-specific constraints
    assert(!(AC_BUS_WORDS > WIN_W) &&
           "ac_window (valid) does not support AC_BUS_WORDS > window width.");
    assert(!(((WIN_W - 1) % AC_BUS_WORDS) != 0) &&
           "ac_window (valid) requires (WIN_W - 1) divisible by AC_BUS_WORDS.");

    static constexpr int d_mult = CONFIG_T::n_filt / CONFIG_T::n_chan;
    static constexpr int ce_reuse_factor =
        CONFIG_T::reuse_factor * (CONFIG_T::strategy == nnet::latency);
    (void)ce_reuse_factor;

    const ac_int<ac::nbits<IMG_W>::val,  false> width  = IMG_W;
    const ac_int<ac::nbits<IMG_H>::val, false> height = IMG_H;

    constexpr int AC_WORDS = WINDOW_2D_TYPE::AC_WORDS;
    ac_array<in_vector_t, CONFIG_T::filt_height, AC_WORDS> window_out;

    bool sof_out = false, sol_out = false, eol_out = false;
    bool eof_out = false;
    ac_int<AC_BUS_WORDS, false> vld_out = 0;

    data_T din;
    res_T  res_pack;
    out_vector_t out_vec;

    typename CONFIG_T::accum_t acc[CONFIG_T::n_filt];
    out_base_t res_out[CONFIG_T::n_filt];

    bool in_read = true;

    constexpr int iterations __attribute__((unused)) =
        (IMG_W * IMG_H) / AC_BUS_WORDS;

    #pragma hls_iterations iterations
    #pragma hls_pipeline_init_interval 1
    do {
      din = data.read();

      window.run(din, width, height, in_read,
                 window_out, sof_out, eof_out, sol_out, eol_out, vld_out);

      if (vld_out) {
        res_pack = 0;

        #pragma hls_unroll yes
        for (int m = 0; m < AC_BUS_WORDS; m++) {

          #pragma hls_unroll
          for (unsigned jj = 0; jj < CONFIG_T::n_chan; jj++) {

            #pragma hls_unroll
            for (unsigned kk = 0; kk < d_mult; kk++) {
              const int filt_idx = jj * d_mult + kk;

              acc[filt_idx] = (typename CONFIG_T::accum_t)biases[filt_idx];

              #pragma hls_unroll
              for (unsigned ii = 0; ii < CONFIG_T::kernel_size; ii++) {
                const unsigned wi = ii / CONFIG_T::filt_width;
                const unsigned wj = ii - wi * CONFIG_T::filt_width;

                const in_base_t pix = in_base_t(window_out[wi][wj + m][jj]);
                const int weight_idx =
                    (ii * CONFIG_T::n_chan + jj) * d_mult + kk;

                acc[filt_idx] += CONFIG_T::mult_config::template product<
                    in_base_t, typename CONFIG_T::mult_config::weight_t>::product(
                    pix, weights[weight_idx]);
              }

              res_out[filt_idx] =
                  cast<in_base_t, out_base_t, typename CONFIG_T::mult_config>(acc[filt_idx]);
            }
          }

          #pragma hls_unroll yes
          for (unsigned ff = 0; ff < CONFIG_T::n_filt; ff++) {
            out_vec[ff] = res_out[ff];
          }

          res_pack[m] = out_vec;
        }

        res.write(res_pack);
      }

    } while (!eof_out);
  }

private:
  WINDOW_2D_TYPE window; // keep as member (consistent w/ class-based style)
};

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_window_dispatch_cl(
    ac_channel<data_T> &data,
    ac_channel<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]) {

  if constexpr (CONFIG_T::padding == padding_type::same) {
    static depthwise_conv2d_window_v2_flush_same_cl<data_T, res_T, CONFIG_T> impl_same;
    impl_same.run(data, res, weights, biases);
  } else if constexpr (CONFIG_T::padding == padding_type::valid) {
    static depthwise_conv2d_window_v2_valid_cl<data_T, res_T, CONFIG_T> impl_valid;
    impl_valid.run(data, res, weights, biases);
  } else {
    static_assert(CONFIG_T::padding == padding_type::same ||
                  CONFIG_T::padding == padding_type::valid,
                  "Unsupported padding type for depthwise_conv_2d_window_dispatch_cl");
  }
}

} // namespace nnet

#endif
