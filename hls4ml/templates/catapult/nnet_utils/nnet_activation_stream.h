
// Change History:
//   2022-06-30  dgburnette - Cleaned up code to separate AC Math from LUT code.
//                            Activation functions not implemented in AC Math will assert.
//   2022-06-28  dgburnette - Replaced AP Types with AC Datatypes.

#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_stream.h"
#include "nnet_types.h"
#include <ac_channel.h>
#include <ac_fixed.h>
#include <ac_math/ac_elu_pwl.h>
#include <ac_math/ac_pow_pwl.h>
#include <ac_math/ac_relu.h>
#include <ac_math/ac_selu_pwl.h>
#include <ac_math/ac_sigmoid_pwl.h>
#include <ac_math/ac_softmax_pwl_new.h>
#include <ac_math/ac_softplus_pwl.h>
#include <ac_math/ac_softsign_pwl.h>
#include <ac_math/ac_tanh_pwl.h>
#include <ac_std_float.h>
#include <cmath>

namespace nnet {

// *************************************************
//       LINEAR Activation
// *************************************************
// Adding this to work around problem with Catapult and SR model where the output channel appears to be inout
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void linear(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    LinearActLoop: for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        LinearPackLoop: for (unsigned int j = 0; j < res_T::size; j++) {
            out_data[j] = in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       RELU Activation
// *************************************************
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void relu(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    ReLUActLoop: for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        ReLUPackLoop: for (unsigned int j = 0; j < res_T::size; j++) {
            ac_math::ac_relu(in_data[j], out_data[j]);
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    SigmoidActLoop: for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        SigmoidPackLoop: for (unsigned int j = 0; j < res_T::size; j++) {
            // ac_math::ac_sigmoid_pwl(in_data[j], out_data[j]);
            ac_sigmoid_pwl_wrapper(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void softmax(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    static_assert(CONFIG_T::axis == -1, "Softmax layer only supports axis=-1.");
    
    typename data_T::value_type data_cache[data_T::size];
    typename res_T::value_type res_cache[res_T::size];
    #pragma hls_pipeline_init_interval 1
    SoftmaxInitLoop: for (unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();

        #pragma hls_unroll
        SoftmaxInitPackLoop: for (unsigned j = 0; j < data_T::size; j++) {
            data_cache[j] = in_pack[j];
        }

        res_T out_pack;
        ac_softmax_pwl_wrapper(data_cache, res_cache);

        #pragma hls_unroll
        SoftmaxResPackLoop: for (unsigned j = 0; j < res_T::size; j++) {
            out_pack[j] = res_cache[j];
        }

        res.write(out_pack);
    }
}

// *************************************************
//       TanH Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    TanHActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        TanHPackLoop: for (int j = 0; j < res_T::size; j++) {
            // int data_round = in_data[j]*CONFIG_T::table_size/8;
            ac_math::ac_tanh_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void hard_sigmoid(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    // typename data_T::value_type slope = (typename data_T::value_type)0.2;
    // typename data_T::value_type shift = (typename data_T::value_type)0.5;

    #pragma hls_pipeline_init_interval 1
    HardSigmoidActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        HardSigmoidPackLoop: for (int j = 0; j < res_T::size; j++) {
            typename data_T::value_type datareg = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (datareg > 1)
                datareg = 1;
            else if (datareg < 0)
                datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Hard TanH Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void hard_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
// typename data_T::value_type slope = (typename data_T::value_type) 0.2;
// typename data_T::value_type shift = (typename data_T::value_type) 0.5;

    #pragma hls_pipeline_init_interval 1
    HardTanhActLoop: for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        HardTanhPackLoop: for (unsigned int j = 0; j < res_T::size; j++) {
            auto sigmoid = CONFIG_T::slope * in_data[j] + CONFIG_T::shift;
            if (sigmoid > 1)
                sigmoid = 1;
            else if (sigmoid < 0)
                sigmoid = 0;
            out_data[j] = 2 * sigmoid - 1;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************




// -------------------------------------------------
// General version: For non-packed data_T
// -------------------------------------------------

enum activation_impl_type {ac_window,non_packed};

#pragma hls_design block
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu_nonpacked(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    LeakyReLUActLoop: for (unsigned int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        LeakyReLUPackLoop: for (unsigned int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha * in_data[j];
        }
        res.write(out_data);
    }
        // std::cout << "INFO: leaky_relu generic path (non-packed) selected\n";

}


// -------------------------------------------------
// Specialized version: For ac_array<ac_packed_vector<T, N>, B>
// -------------------------------------------------
#pragma hls_design block
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu_packed(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) 
{

    constexpr int BUS_WORDS = data_T::dim1;
    data_T in_array;
    typedef typename data_T::ElemType in_vector_t;
    in_vector_t in_vector;
    constexpr int N_CHANNELS_IN = in_vector_t::packed_words;

    typedef typename res_T::ElemType out_vector_t;
    out_vector_t out_vector;
    res_T out_array;
    constexpr int N_CHANNELS_OUT = out_vector_t::packed_words;

    typedef typename in_vector_t::base_type in_base_t;
    typedef typename out_vector_t::base_type out_base_t;
    static_assert(N_CHANNELS_IN == N_CHANNELS_OUT, "Input/output channel mismatch!");

    // std::cout << "INFO: BUS_WORDS = " << BUS_WORDS << std::endl;
    // std::cout << "INFO: N_CHANNELS_IN = " << N_CHANNELS_IN << std::endl;
    // std::cout << "INFO: N_CHANNELS_IN = " << N_CHANNELS_IN << std::endl;

    #pragma hls_pipeline_init_interval 1
    LeakyReLUActLoop: for (unsigned int i = 0; i <CONFIG_T::n_in/(BUS_WORDS*N_CHANNELS_IN); i++) {
        in_array = data.read();
        #pragma hls_unroll
        BusLoop: for (int b = 0; b < BUS_WORDS; b++) {
            in_vector=in_array[b];
            #pragma hls_unroll
            VecLoop: for (int v = 0; v < N_CHANNELS_IN; v++) {
                in_base_t val_unpacked = in_vector[v];
                in_base_t result;
                if (val_unpacked > 0)
                    result = val_unpacked;
                else
                    result = in_base_t(alpha * val_unpacked);
                out_vector[v] = result;
            }
            out_array[b]=out_vector;
        }
        res.write(out_array);
    }
    // std::cout << "INFO: leaky_relu packed path (ac_packed_vector) selected\n";
}

template <class data_T, class param_T, class res_T, typename CONFIG_T>
void leaky_relu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) {
    if constexpr (CONFIG_T::impl == activation_impl_type::ac_window) {
        leaky_relu_packed<data_T, param_T, res_T, CONFIG_T>(data, alpha, res);
    } else {
        leaky_relu_nonpacked<data_T, param_T, res_T, CONFIG_T>(data, alpha, res);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************

#pragma hls_design block
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void thresholded_relu(ac_channel<data_T> &data, param_T theta, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    ThresholdedReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        ThresholdedReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > theta)
                out_data[j] = in_data[j];
            else
                out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void softplus(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    SoftplusActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        SoftplusPackLoop: for (int j = 0; j < res_T::size; j++) {
            ac_softplus_pwl_wrapper(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void softsign(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    SoftsignActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        SoftsignPackLoop: for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_softsign_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************

#pragma hls_design block
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void elu(ac_channel<data_T> &data, param_T alpha, ac_channel<res_T> &res) 
{
    EluActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        EluPackLoop: for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_elu_pwl(in_data[j], out_data[j], alpha);
        }
        res.write(out_data);
    }
}

// *************************************************
//       SELU Activation
// *************************************************

#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void selu(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    SeluActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;
        #pragma hls_unroll
        SeluPackLoop: for (int j = 0; j < res_T::size; j++) {
            ac_math::ac_selu_pwl(in_data[j], out_data[j]);
        }
        res.write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
#pragma hls_design block
template <class data_T, class param_T, class res_T, typename CONFIG_T>
void prelu(ac_channel<data_T> &data, param_T alpha[CONFIG_T::n_in], ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    PReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        PReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0)
                out_data[j] = in_data[j];
            else
                out_data[j] = alpha[i * res_T::size + j] * in_data[j];
        }
        res.write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void binary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    using cache_T = ac_int<2, true>;
    #pragma hls_pipeline_init_interval 1
    PReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        cache_T cache;
        res_T out_data;

        #pragma hls_unroll
        PReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] >= 0)
                cache = 1;
            else
                cache = -1;

            out_data[j] = binary_cast<cache_T, typename res_T::value_type>(cache);
        }
        res.write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T> 
void ternary_tanh(ac_channel<data_T> &data, ac_channel<res_T> &res) 
{
    #pragma hls_pipeline_init_interval 1
    PReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        #pragma hls_unroll
        PReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 1)
                out_data[j] = (typename res_T::value_type)1;
            else if (in_data[j] <= -1)
                out_data[j] = (typename res_T::value_type) - 1;
            else
                out_data[j] = (typename res_T::value_type)0;
        }
        res.write(out_data);
    }
}

} // namespace nnet

#endif

