#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) 
{
    constexpr unsigned mult_n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned block_factor = DIV_ROUNDUP(mult_n_in * mult_n_out, CONFIG_T::reuse_factor);

    constexpr unsigned multscale = block_factor / mult_n_out;

    assert((block_factor % mult_n_out == 0 || CONFIG_T::reuse_factor >= mult_n_in) &&
           "The current Reuse Factor is not allowed");
    assert((CONFIG_T::reuse_factor <= CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan) &&
           "This function is correct only for RF <= FILT_HEIGHT * FILT_WIDTH * N_CHAN");

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][mult_n_out];

    //#pragma hls_unroll // We don't want this loop unrolled
    PartitionLoop: for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

        #pragma hls_unroll
        PixelInitAccumLoop: for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {

            #pragma hls_unroll
            InitAccumLoop: for (unsigned i_acc = 0; i_acc < mult_n_out; i_acc++) {
                acc[i_pxl][i_acc] = (typename CONFIG_T::accum_t)biases[i_acc];
            }
        }

        #pragma hls_pipeline_init_interval 1
        ReuseLoop: for (unsigned i_rf = 0; i_rf < CONFIG_T::reuse_factor; i_rf++) {
            unsigned i_w = i_rf;
            unsigned i_in = i_rf;
            unsigned i_out = 0;
            unsigned i_acc = 0;

            #pragma hls_unroll
            MultLoop: for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                #pragma hls_unroll
                PixelMultLoop: for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
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
        PixelResultLoop: for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            // Cast to "res_t" type
            #pragma hls_unroll
            ResultLoop: for (unsigned i_res = 0; i_res < mult_n_out; i_res++) {
                *(res++) = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_pxl][i_res]);
            }
        }
    }
}

} // namespace nnet
#endif
