#ifndef NNET_NORMALIZE_STREAM_H_
#define NNET_NORMALIZE_STREAM_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include <ac_channel.h>

namespace nnet {

struct normalize_config {
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 1;
};

//// this implementation is only for the case where we have n_chan = 3
//// no Catapult component in the library error when fixed<64, 32> is set as the data_T

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void normalize_mean(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream) {

    typename data_T::value_type rgb_mean[3] = {0.4488 * 255, 0.4371 * 255, 0.4040 * 255};

#pragma hls_pipeline_init_interval 1
ImageHeight:
    for (unsigned h = 0; h < CONFIG_T::in_height; h++) {
    ImageWidth:
        for (unsigned w = 0; w < CONFIG_T::in_width; w++) {
            data_T pixel = data_stream.read();
            res_T pixel_out;
        #pragma hls_unroll
        ImageChan:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                pixel_out[c] = (pixel[c] - rgb_mean[c]) / (typename data_T::value_type)127.5;
            }
            res_stream.write(pixel_out);
        }
    }
}

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void denormalize_mean(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream) {

    typename data_T::value_type rgb_mean[3] = {0.4488 * 255, 0.4371 * 255, 0.4040 * 255};

#pragma hls_pipeline_init_interval 1
ImageHeight:
    for (unsigned h = 0; h < CONFIG_T::in_height; h++) {
    ImageWidth:
        for (unsigned w = 0; w < CONFIG_T::in_width; w++) {
            data_T pixel = data_stream.read();
            res_T pixel_out;
        #pragma hls_unroll
        ImageChan:
            for (unsigned c = 0; c < CONFIG_T::n_chan; c++) {
                pixel_out[c] = pixel[c] * ((typename data_T::value_type)127.5) + rgb_mean[c];
            }
            res_stream.write(pixel_out);
        }
    }
}

} // namespace nnet
#endif
