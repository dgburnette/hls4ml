#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include <ac_channel.h>

namespace nnet {

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void embedding(ac_channel<data_T> &data, ac_channel<res_T> &res,
               typename CONFIG_T::embeddings_t embeddings[CONFIG_T::vocab_size * CONFIG_T::n_out]) {
    data_T in_data = data.read();
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
#pragma hls_pipeline_init_interval ce_reuse_factor
InputSequence:
    for (int j = 0; j < data_T::size; j++) {

        res_T res_pack;

    #pragma hls_unroll
    DenseEmbedding:
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            res_pack[i] = embeddings[in_data[j] * CONFIG_T::n_out + i];
        }
        res.write(res_pack);
    }
}

} // namespace nnet

#endif
