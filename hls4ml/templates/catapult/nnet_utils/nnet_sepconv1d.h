#ifndef NNET_SEPARABLE_CONV1D_H_
#define NNET_SEPARABLE_CONV1D_H_

#include "nnet_common.h"
#include "nnet_conv1d.h"
#include "nnet_sepconv1d_latency.h"
//#include "nnet_sepconv1d_resource.h"
#include <cstdlib>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                          res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                          const typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan],
                          const typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {
    if (CONFIG_T::strategy == nnet::latency) {
        depthwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    } else {
        assert("Resource strategy for DepthwiseConv1D is not supported." && false);
    }
}

template <class data_T, class dw_res_T, class res_T, typename CONFIG_T>
void separable_conv_1d_cl(
    data_T data[CONFIG_T::depthwise_config::in_width * CONFIG_T::depthwise_config::n_chan],
    res_T res[CONFIG_T::pointwise_config::out_width * CONFIG_T::pointwise_config::n_filt],
    const typename CONFIG_T::depthwise_config::weight_t
        depthwise_weights[CONFIG_T::depthwise_config::filt_width * CONFIG_T::depthwise_config::n_chan],
    const typename CONFIG_T::pointwise_config::weight_t
        pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
    const typename CONFIG_T::depthwise_config::bias_t depthwise_biases[CONFIG_T::depthwise_config::n_chan],
    const typename CONFIG_T::pointwise_config::bias_t pointwise_biases[CONFIG_T::pointwise_config::n_filt]) {

    dw_res_T depthwise_res[CONFIG_T::depthwise_config::out_width * CONFIG_T::depthwise_config::n_filt];

    depthwise_conv_1d_cl<data_T, dw_res_T, typename CONFIG_T::depthwise_config>(data, depthwise_res, depthwise_weights,
                                                                                depthwise_biases);
    pointwise_conv_1d_cl<dw_res_T, res_T, typename CONFIG_T::pointwise_config>(depthwise_res, res, pointwise_weights,
                                                                               pointwise_biases);
}

} // namespace nnet

#endif
