#ifndef NNET_IMAGE_STREAM_H_
#define NNET_IMAGE_STREAM_H_

#include "nnet_common.h"
#include <ac_channel.h>

namespace nnet {

#pragma hls_design block
template <class data_T, typename CONFIG_T> void resize_nearest(ac_channel<data_T> &image, ac_channel<data_T> &resized) {
    assert(CONFIG_T::new_height % CONFIG_T::height == 0);
    assert(CONFIG_T::new_width % CONFIG_T::width == 0);
    constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
    constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;

#pragma hls_pipeline_init_interval 1
ImageHeight:
    for (unsigned h = 0; h < CONFIG_T::height; h++) {
        //#pragma HLS PIPELINE

        data_T data_in_row[CONFIG_T::width];

    // #pragma hls_unroll
    ImageWidth:
        for (unsigned i = 0; i < CONFIG_T::width; i++) {
            //#pragma HLS UNROLL

            data_T in_data = image.read();

        #pragma hls_unroll
        ImageChan:
            for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
                //#pragma HLS UNROLL

                data_in_row[i][j] = in_data[j];
            }
        }

    #pragma hls_unroll
    ResizeHeight:
        for (unsigned i = 0; i < ratio_height; i++) {
        //#pragma HLS UNROLL

        #pragma hls_unroll
        ImageWidth2:
            for (unsigned l = 0; l < CONFIG_T::width; l++) {
            //#pragma HLS UNROLL

            // #pragma hls_unroll
            ResizeWidth:
                for (unsigned j = 0; j < ratio_width; j++) {
                    //#pragma HLS UNROLL

                    data_T out_data;
                //#pragma HLS DATA_PACK variable=out_data

                #pragma hls_unroll
                ResizeChan:
                    for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                        //#pragma HLS UNROLL

                        out_data[k] = data_in_row[l][k];
                    }

                    resized.write(out_data);
                }
            }
        }
    }
}

} // namespace nnet

#endif
