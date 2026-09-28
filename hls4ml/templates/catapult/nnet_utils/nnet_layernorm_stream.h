#ifndef NNET_LAYERNORM_STREAM_H_
#define NNET_LAYERNORM_STREAM_H_

#include "nnet_common.h"
#include "nnet_layernorm.h"
#include "nnet_types.h"
#include <ac_channel.h>

namespace nnet {

// ****************************************************
//       Streaming Layer Normalization
// ****************************************************
#pragma hls_design block

template <class data_T, class res_T, typename CONFIG_T>
void layernormalize(ac_channel<data_T> &data, ac_channel<res_T> &res,
                    typename CONFIG_T::scale_t scale[CONFIG_T::n_in / CONFIG_T::seq_len],
                    typename CONFIG_T::bias_t bias[CONFIG_T::n_in / CONFIG_T::seq_len])
{
    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;

    typename data_T::value_type in_val[dim];
    typename res_T::value_type outval[dim];

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
LAYERNORM_SEQ_LOOP:
    for (unsigned int j = 0; j < CONFIG_T::n_in / data_T::size; j++) {
        data_T in_pack = data.read();
        #pragma hls_unroll
        LAYERNORM_LOAD:
        for (unsigned int i = 0; i < data_T::size; i++) {
            in_val[i] = in_pack[i];
        }

        layernorm_1d<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(in_val, outval, scale, bias);

        res_T out_pack;
        #pragma hls_unroll
        LAYERNORM_STORE:
        for (unsigned int i = 0; i < res_T::size; i++) {
            out_pack[i] = outval[i];
        }
        res.write(out_pack);
    }
}

} // namespace nnet

#endif
