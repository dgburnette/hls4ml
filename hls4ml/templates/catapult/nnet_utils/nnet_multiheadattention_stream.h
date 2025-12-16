#ifndef NNET_MHT_H_
#define NNET_MHT_H_

// Include necessary headers
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_mult.h"
#include <ac_channel.h>
#include <ac_math.h>

namespace nnet {

// Configuration structure for multi-head attention
struct multiheadattention_config {
    typedef float bias_t;                  // Data type for biases
    typedef float weight_t;                // Data type for weights
    typedef float accum_t;                 // Data type for accumulators
    typedef ac_fixed<16, 8, true> multi_t; // Fixed-point type for intermediate computations

    static const unsigned num_heads = 10;      // Number of attention heads
    static const unsigned head_dim_key = 10;   // Dimension of key vectors per head
    static const unsigned head_dim_value = 10; // Dimension of value vectors per head
    static const unsigned feature_dim = 20;    // Input feature dimension
    static const unsigned seq_len = 500;       // Sequence length

    static const unsigned io_type = io_stream;     // I/O type
    static const unsigned strategy = latency;        // Strategy for optimization
    static const unsigned reuse_factor = 1;          // Reuse factor for hardware optimization
    static const bool store_weights_in_bram = false; // Whether to store weights in BRAM

    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>; // Multiplication template
};

// Data structure for packing multiple elements
template <int PackSize, class data_T> struct datapack {
    data_T data[PackSize];

    datapack() {}

    datapack(data_T p[PackSize]) {
        for (unsigned i = 0; i < PackSize; i++) {
            data[i] = p[i];
        }
    }
};

// Function to read data from an array of streams into an array
template <class data_T, unsigned num_heads, unsigned feature_dim> 
void read_stream_array(ac_channel<data_T> data_fifo[num_heads][feature_dim], unsigned head_idx, data_T data_buffer[feature_dim]) 
{
    read_stream_array_loop: for (unsigned feature_idx = 0; feature_idx < feature_dim; ++feature_idx) {
        data_buffer[feature_idx] = data_fifo[head_idx][feature_idx].read();
    }
}

#include <ac_math/ac_inverse_sqrt_pwl.h>
using namespace ac_math;

// Function to prepare input data for processing
#pragma hls_design block
template <class data_T, typename CONFIG_T>
void data_preparation(ac_channel<data_T> &query_data, 
                      ac_channel<data_T> &key_value_data,
                      ac_channel<typename data_T::value_type> query_fifo[CONFIG_T::num_heads][CONFIG_T::feature_dim],
                      ac_channel<typename data_T::value_type> key_value_fifo[CONFIG_T::num_heads][CONFIG_T::feature_dim]) 
{
    // Iterate over the sequence and feature dimensions to write data into FIFOs
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    data_preparation_seq_loop: for (unsigned seq_idx = 0; seq_idx < CONFIG_T::seq_len; ++seq_idx) {
        data_T data_q = query_data.read();
        data_T data_k = key_value_data.read();
        data_preparation_feature_loop: for (unsigned feature_idx = 0; feature_idx < CONFIG_T::feature_dim; ++feature_idx) {
            data_preparation_head_loop: for (unsigned head_idx = 0; head_idx < CONFIG_T::num_heads; ++head_idx) {    
                query_fifo[head_idx][feature_idx].write(data_q[feature_idx]);
                key_value_fifo[head_idx][feature_idx].write(data_k[feature_idx]);
            }
        }
    }
}

// Function to perform linear projection for query, key, and value
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void linear_projection(ac_channel<data_T> query_fifo[CONFIG_T::num_heads][CONFIG_T::feature_dim],
                       ac_channel<data_T> key_value_fifo[CONFIG_T::num_heads][CONFIG_T::feature_dim],
                       ac_channel<datapack<CONFIG_T::head_dim_key, res_T>> projected_query_fifo[CONFIG_T::num_heads],
                       ac_channel<datapack<CONFIG_T::head_dim_key, res_T>> projected_key_fifo[CONFIG_T::num_heads],
                       ac_channel<datapack<CONFIG_T::head_dim_value, res_T>> projected_value_fifo[CONFIG_T::num_heads],
                       typename CONFIG_T::weight_t key_weights[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key],
                       typename CONFIG_T::bias_t key_biases[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
                       typename CONFIG_T::weight_t query_weights[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key],
                       typename CONFIG_T::bias_t query_biases[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
                       typename CONFIG_T::weight_t value_weights[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_value],
                       typename CONFIG_T::bias_t value_biases[CONFIG_T::num_heads * CONFIG_T::head_dim_value])
{
    // Iterate over the sequence length to process each token
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    linear_projection_seq_loop: for (unsigned seq_idx = 0; seq_idx < CONFIG_T::seq_len; ++seq_idx) {
        // std::cout << "Token # " << seq_idx + 1 << "/" << CONFIG_T::seq_len << " (= seq_len)" << std::endl;
        linear_projection_head_loop: for (unsigned head_idx = 0; head_idx < CONFIG_T::num_heads; ++head_idx) {
            // Input data buffers
            static data_T query_buffer[CONFIG_T::feature_dim];
            static data_T key_value_buffer[CONFIG_T::feature_dim];

            // Output data buffers
            res_T projected_query_buffer[CONFIG_T::head_dim_key];
            res_T projected_key_buffer[CONFIG_T::head_dim_key];
            res_T projected_value_buffer[CONFIG_T::head_dim_value];

            read_stream_array<data_T, CONFIG_T::num_heads, CONFIG_T::feature_dim>(query_fifo, head_idx, query_buffer);
            read_stream_array<data_T, CONFIG_T::num_heads, CONFIG_T::feature_dim>(key_value_fifo, head_idx, key_value_buffer);

            dense<data_T, res_T, typename CONFIG_T::config_mult1>(query_buffer, projected_query_buffer, 
                                                            query_weights + (CONFIG_T::head_dim_key * CONFIG_T::feature_dim * head_idx),
                                                            query_biases + (CONFIG_T::head_dim_key * head_idx));
            dense<data_T, res_T, typename CONFIG_T::config_mult1>(key_value_buffer, projected_key_buffer, 
                                                            key_weights + (CONFIG_T::head_dim_key * CONFIG_T::feature_dim * head_idx),
                                                            key_biases + (CONFIG_T::head_dim_key * head_idx));
            dense<data_T, res_T, typename CONFIG_T::config_mult1>(key_value_buffer, projected_value_buffer, 
                                                            value_weights + (CONFIG_T::head_dim_value * CONFIG_T::feature_dim * head_idx),
                                                            value_biases + (CONFIG_T::head_dim_value * head_idx));

            datapack<CONFIG_T::head_dim_key, res_T> projected_query_pack = datapack<CONFIG_T::head_dim_key, res_T>(projected_query_buffer);
            datapack<CONFIG_T::head_dim_key, res_T> projected_key_pack = datapack<CONFIG_T::head_dim_key, res_T>(projected_key_buffer);
            datapack<CONFIG_T::head_dim_value, res_T> projected_value_pack = datapack<CONFIG_T::head_dim_value, res_T>(projected_value_buffer);
            
            projected_query_fifo[head_idx].write(projected_query_pack);
            projected_key_fifo[head_idx].write(projected_key_pack);
            projected_value_fifo[head_idx].write(projected_value_pack);
        }
    }
}

// Function to compute attention scores (scaled dot-product attention)
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void compute_attention_scores(ac_channel<datapack<CONFIG_T::head_dim_key, data_T>> projected_query_fifo[CONFIG_T::num_heads],
                              ac_channel<datapack<CONFIG_T::head_dim_key, data_T>> projected_key_fifo[CONFIG_T::num_heads],
                              ac_channel<res_T> attention_scores[CONFIG_T::num_heads]) 
{
    bool fifo_read = true;
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    compute_attention_scores_seq_loop: for (unsigned seq_q_idx = 0; seq_q_idx < CONFIG_T::seq_len; ++seq_q_idx) {
        attention_head_loop: for (unsigned head_idx = 0; head_idx < CONFIG_T::num_heads; ++head_idx) { 
            data_T dk;
            typename CONFIG_T::inverse_sqrt_in_t inverse_sqrt_in = CONFIG_T::head_dim_key;
            typename CONFIG_T::inverse_sqrt_out_t inverse_sqrt_out;
            ac_inverse_sqrt_pwl(inverse_sqrt_in, inverse_sqrt_out);
            dk = inverse_sqrt_out;

            data_T product_result;
            typename CONFIG_T::accum_t score_accumulator;
            static data_T query_buffer[CONFIG_T::head_dim_key];
            static data_T key_buffer[CONFIG_T::num_heads][CONFIG_T::seq_len * CONFIG_T::head_dim_key];
            data_T product_buffer[CONFIG_T::seq_len];
            res_T softmax_output[CONFIG_T::seq_len];

            datapack<CONFIG_T::head_dim_key, data_T> key_pack, query_pack;

            if (fifo_read) {
                read_key_loop: for (unsigned seq_k_idx = 0; seq_k_idx < CONFIG_T::seq_len; ++seq_k_idx) {
                    key_pack = projected_key_fifo[head_idx].read();
                    #pragma hls_unroll yes
                    load_key_loop: for (unsigned key_idx = 0; key_idx < CONFIG_T::head_dim_key; ++key_idx) {
                        key_buffer[head_idx][seq_k_idx * CONFIG_T::head_dim_key + key_idx] = key_pack.data[key_idx];
                    }
                }
            }
            
            query_pack = projected_query_fifo[head_idx].read();
            #pragma hls_unroll yes
            load_query_loop: for (unsigned query_idx = 0; query_idx < CONFIG_T::head_dim_key; ++query_idx) {
                query_buffer[query_idx] = query_pack.data[query_idx];
            }

            compute_product_loop: for (unsigned seq_k_idx = 0; seq_k_idx < CONFIG_T::seq_len; ++seq_k_idx) {
                score_accumulator = 0;
                dot_product_loop: for (unsigned key_idx = 0; key_idx < CONFIG_T::head_dim_key; ++key_idx) {
                    product_result = CONFIG_T::template product<data_T, data_T>::product(
                        query_buffer[key_idx], key_buffer[head_idx][seq_k_idx * CONFIG_T::head_dim_key + key_idx]);
                    score_accumulator += product_result;
                }
                product_buffer[seq_k_idx] = score_accumulator * dk;
            }

            softmax<data_T, res_T, typename CONFIG_T::softmax_config1>(product_buffer, softmax_output);

            write_scores_loop: for (unsigned seq_k_idx = 0; seq_k_idx < CONFIG_T::seq_len; ++seq_k_idx) {
                attention_scores[head_idx].write(softmax_output[seq_k_idx]);
            }
        }
        fifo_read = false;
    }
}

// Function to apply attention scores to value vectors
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void apply_attention_scores_to_values(ac_channel<res_T> attention_scores[CONFIG_T::num_heads],
                                      ac_channel<datapack<CONFIG_T::head_dim_value, data_T>> projected_value_fifo[CONFIG_T::num_heads],
                                      ac_channel<res_T> attention_output_fifo[CONFIG_T::num_heads][CONFIG_T::head_dim_value]) 
{
    bool fifo_read = true;
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    apply_attention_scores_seq_loop: for (unsigned seq_q_idx = 0; seq_q_idx < CONFIG_T::seq_len; ++seq_q_idx) { 
        apply_head_loop: for (unsigned head_idx = 0; head_idx < CONFIG_T::num_heads; ++head_idx) {   
            static data_T value_buffer[CONFIG_T::num_heads][CONFIG_T::seq_len * CONFIG_T::head_dim_value];
            datapack<CONFIG_T::head_dim_value, data_T> value_pack;
        
            // Read projected values into a buffer
            if (fifo_read) {
                read_val_loop: for (unsigned seq_k_idx = 0; seq_k_idx < CONFIG_T::seq_len; ++seq_k_idx) {
                    value_pack = projected_value_fifo[head_idx].read();
                    #pragma hls_unroll yes
                    load_value_loop: for (unsigned value_idx = 0; value_idx < CONFIG_T::head_dim_value; ++value_idx) {
                        value_buffer[head_idx][seq_k_idx + CONFIG_T::seq_len * value_idx] = value_pack.data[value_idx];
                    }
                }
            }

            data_T weighted_sum, product_result;
            static data_T attention_score_row[CONFIG_T::seq_len];
            load_attention_scores_loop: for (unsigned seq_k_idx = 0; seq_k_idx < CONFIG_T::seq_len; ++seq_k_idx) {
                attention_score_row[seq_k_idx] = attention_scores[head_idx].read();
            }
            
            compute_weighted_sum_loop: for (unsigned value_idx = 0; value_idx < CONFIG_T::head_dim_value; ++value_idx) {
                weighted_sum = 0;
                dot_product_loop: for (unsigned seq_k_idx = 0; seq_k_idx < CONFIG_T::seq_len; ++seq_k_idx) {
                    product_result = CONFIG_T::template product<data_T, data_T>::product(
                        attention_score_row[seq_k_idx], value_buffer[head_idx][seq_k_idx + CONFIG_T::seq_len * value_idx]);
                    weighted_sum += product_result;
                }
                attention_output_fifo[head_idx][value_idx].write(weighted_sum);
            }
        }
        fifo_read = false;
    }
}

// Function to combine outputs from all attention heads
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void project_and_combine_heads(ac_channel<data_T> attention_output_fifo[CONFIG_T::num_heads][CONFIG_T::head_dim_value],
                               ac_channel<res_T> &attention_output,
                               typename CONFIG_T::weight_t attention_output_weights[CONFIG_T::num_heads * CONFIG_T::head_dim_value * CONFIG_T::feature_dim],
                               typename CONFIG_T::bias_t attention_output_biases[CONFIG_T::feature_dim]) 
{
    static data_T concatenated_heads[CONFIG_T::num_heads * CONFIG_T::head_dim_value];
    typename res_T::value_type dense_output[CONFIG_T::feature_dim];
    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor;
    (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    project_and_combine_heads_seq_loop: for (unsigned seq_idx = 0; seq_idx < CONFIG_T::seq_len; ++seq_idx) {
        // Read and concatenate attention outputs from all heads
        read_heads_loop: for (unsigned head_idx = 0; head_idx < CONFIG_T::num_heads; ++head_idx) {
            read_head_values_loop: for (unsigned value_idx = 0; value_idx < CONFIG_T::head_dim_value; ++value_idx) {
                concatenated_heads[head_idx * CONFIG_T::head_dim_value + value_idx] = 
                    attention_output_fifo[head_idx][value_idx].read();
            }
        }

        // Apply dense layer to combine heads
        dense<data_T, typename res_T::value_type, typename CONFIG_T::config_mult2>(concatenated_heads, dense_output, attention_output_weights,
                                                                   attention_output_biases);

        // Write the combined output
        res_T attention; 
        write_output_loop: for (unsigned feature_idx = 0; feature_idx < CONFIG_T::feature_dim; ++feature_idx) {
            attention[feature_idx] = dense_output[feature_idx];
        }
        attention_output.write(attention);
    }
}


// Main multi-head attention function
#pragma hls_design block
template <class data_T, class res_T, typename CONFIG_T>
void multiheadattention(
    ac_channel<data_T> &query_data, 
    ac_channel<data_T> &key_value_data, 
    ac_channel<res_T> &attention_output,
    typename CONFIG_T::weight_t
        attention_output_weights[CONFIG_T::num_heads * CONFIG_T::head_dim_value * CONFIG_T::feature_dim],    
    typename CONFIG_T::bias_t attention_output_biases[CONFIG_T::feature_dim],
    typename CONFIG_T::weight_t key_weights[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::bias_t key_biases[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t query_weights[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::bias_t query_biases[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t value_weights[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_value],
    typename CONFIG_T::bias_t value_biases[CONFIG_T::num_heads * CONFIG_T::head_dim_value]) 
{ 
    #pragma hls_fifo_depth 1
    static ac_channel<typename data_T::value_type> query_fifo[CONFIG_T::num_heads][CONFIG_T::feature_dim];
    #pragma hls_fifo_depth 1
    static ac_channel<typename data_T::value_type> key_value_fifo[CONFIG_T::num_heads][CONFIG_T::feature_dim];
    nnet::data_preparation<data_T, CONFIG_T>(query_data, key_value_data, query_fifo, key_value_fifo);
    
    constexpr int fifo_depth = CONFIG_T::seq_len;
    #pragma hls_fifo_depth fifo_depth
    static ac_channel<datapack<CONFIG_T::head_dim_key, typename res_T::value_type>> projected_query_fifo[CONFIG_T::num_heads];
    #pragma hls_fifo_depth fifo_depth
    static ac_channel<datapack<CONFIG_T::head_dim_key, typename res_T::value_type>> projected_key_fifo[CONFIG_T::num_heads];
    #pragma hls_fifo_depth fifo_depth
    static ac_channel<datapack<CONFIG_T::head_dim_value, typename res_T::value_type>> projected_value_fifo[CONFIG_T::num_heads];
    nnet::linear_projection<typename data_T::value_type, typename res_T::value_type, CONFIG_T>(
        query_fifo, 
        key_value_fifo,
        projected_query_fifo, 
        projected_key_fifo,
        projected_value_fifo,
        key_weights, 
        key_biases, 
        query_weights, 
        query_biases, 
        value_weights, 
        value_biases);
    
    #pragma hls_fifo_depth 1
    static ac_channel<typename res_T::value_type> attention_scores[CONFIG_T::num_heads];
    nnet::compute_attention_scores<typename res_T::value_type, typename res_T::value_type, CONFIG_T>(projected_query_fifo, projected_key_fifo, attention_scores);

    #pragma hls_fifo_depth 1
    static ac_channel<typename res_T::value_type> attention_output_fifo[CONFIG_T::num_heads][CONFIG_T::head_dim_value];
    nnet::apply_attention_scores_to_values<typename res_T::value_type, typename res_T::value_type, CONFIG_T>(attention_scores, projected_value_fifo, attention_output_fifo);
    
    nnet::project_and_combine_heads<typename res_T::value_type, res_T, CONFIG_T>(attention_output_fifo, attention_output,
                                                             attention_output_weights, attention_output_biases);
}

} // namespace nnet

#endif
