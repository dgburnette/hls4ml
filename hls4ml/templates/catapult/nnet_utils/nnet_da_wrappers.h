#ifndef NNET_UNROLLED__H_
#define NNET_UNROLLED__H_

#include "nnet_common.h"
#include "nnet_helpers.h"

namespace nnet {

#pragma hls_design    
template <class data_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::strategy == nnet::distributed_arithmetic, void>::type
dense(ac_channel<data_T> &data_stream, ac_channel<res_T> &res_stream) 
{
    typename data_T::value_type data[CONFIG_T::n_in];

    typename res_T::value_type res[CONFIG_T::n_out];

    if ((CONFIG_T::n_in / data_T::size) > 1) {
        #pragma hls_pipeline_init_interval 1
    }
    DataPrepare: for (unsigned i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
        #pragma hls_unroll
        DataPack: for (unsigned i_pack = 0; i_pack < data_T::size; i_pack++) { 
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    CONFIG_T::dense_da(data, res);

    if ((CONFIG_T::n_out / res_T::size) > 1) {
        #pragma hls_pipeline_init_interval 1
    }
    ResWrite: for (unsigned i_out = 0; i_out < CONFIG_T::n_out / res_T::size; i_out++) {
        res_T res_pack;
        #pragma hls_unroll
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}

#pragma hls_design  
template <typename CONFIG_T, class data_T, class res_T>
typename std::enable_if<CONFIG_T::strategy == nnet::distributed_arithmetic, void>::type
conv1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], res_T res[CONFIG_T::out_width * CONFIG_T::n_filt]) 
{
    constexpr unsigned mult_n_in = CONFIG_T::n_chan * CONFIG_T::filt_width;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];

    res_T out_buf[mult_n_out];

    #pragma hls_pipeline_init_interval 1
    PartitionLoop: for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

        #pragma hls_unroll
        PixelLoop: for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            // Do the matrix-multiply
            CONFIG_T::dense_da(data_buf[i_pxl], out_buf);

            #pragma hls_unroll
            Result: for (int i_res = 0; i_res < mult_n_out; i_res++) {
                *(res++) = out_buf[i_res];
            }
        }
    }
}

template <typename CONFIG_T, class data_T, class res_T>
typename std::enable_if<CONFIG_T::strategy == nnet::distributed_arithmetic, void>::type
conv2d_cl(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
          res_T res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt]) 
{
    constexpr unsigned mult_n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];

    res_T out_buf[mult_n_out];

    #pragma hls_pipeline_init_interval 1
    PartitionLoop: for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

        #pragma hls_unroll
        PixelLoop: for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            data_T cache;

            CONFIG_T::dense_da(data_buf[i_pxl], out_buf);

            #pragma hls_unroll
            Result: for (int i_res = 0; i_res < mult_n_out; i_res++) {
                *(res++) = out_buf[i_res];
            }
        }
    }
}

} // namespace nnet

#endif
