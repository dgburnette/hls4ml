
#ifndef NNET_SCVERIFY_H
#define NNET_SCVERIFY_H

#include <ac_channel.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifndef __SYNTHESIS__

namespace nnet {
    

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>
unsigned int compare_data(std::vector<src_T> src, dst_T dst[SIZE], float threshold = 0.01, bool verbose = false) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;

    unsigned int err_cnt = 0;
    unsigned index = 0;
    for (typename std::vector<src_T>::const_iterator i = in_begin; i != in_end; ++i) {
        double ref = *i;
        dst_T ref_quant(ref);
        double dut = dst[index++].to_double();
        double delta = abs(ref_quant.to_double() - dut);
        if (delta > threshold) {
            std::cout << "Ref " << ref << " Ref(quantized) " << ref_quant << "  DUT " << dut << "     <- MISMATCH"
                      << std::endl;
            err_cnt++;
        } else {
            if (verbose) {
                std::cout << "Ref " << ref << " Ref(quantized) " << ref_quant << "  DUT " << dut << "     <- MISMATCH"
                          << std::endl;
            }
        }
    }
    return err_cnt;
}


template <class src_T, class dst_T, size_t OFFSET, size_t SIZE>
unsigned int compare_data(std::vector<src_T> src, ac_channel<dst_T> &dst, float threshold = 0.01, bool verbose = false) {
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;

    unsigned int err_cnt = 0;
    unsigned index = 0;
    size_t i_pack = 0;
    dst_T dst_pack;
    for (typename std::vector<src_T>::const_iterator i = in_begin; i != in_end; ++i) {
        if (i_pack == 0) {
            dst_pack = dst[index++]; // non-destructive peak of values
        }
        double ref = *i;
        typename dst_T::value_type ref_quant(ref);
        double dut = dst_pack[i_pack++].to_double();
        double delta = abs(ref_quant.to_double() - dut);
        if (delta > threshold) {
            std::cout << "Ref " << ref << " Ref(quantized) " << ref_quant << "  DUT " << dut << "     <- MISMATCH"
                      << std::endl;
            err_cnt++;
        } else {
            if (verbose) {
                std::cout << "Ref " << ref << " Ref(quantized) " << ref_quant << "  DUT " << dut << "     <- MISMATCH"
                          << std::endl;
            }
        }
        if (i_pack == dst_T::size) {
            i_pack = 0;
        }
    }
    return err_cnt;
}

template <class src_T, class dst_T, size_t OFFSET, size_t SIZE, size_t BUS_WORDS>
unsigned int compare_data(std::vector<src_T> src, ac_channel<dst_T> &dst, float threshold = 0.01, bool verbose = false) {
    typedef typename dst_T::ElemType vector_t;
    typedef typename vector_t::base_type base_t;
    constexpr int N_CHANNELS = vector_t::packed_words;

    std::vector<base_t> ref_quantized_data;
    std::vector<base_t> dut_data;

    unsigned int err_cnt = 0;
    unsigned int index = 0;

    // Step 1: Quantize input reference values
    typename std::vector<src_T>::const_iterator in_begin = src.cbegin() + OFFSET;
    typename std::vector<src_T>::const_iterator in_end = in_begin + SIZE;

    for (auto i = in_begin; i != in_end; ++i) {
        double ref = *i;
        base_t ref_quant(ref);
        ref_quantized_data.push_back(ref_quant);
    }

    // Step 2: Peek DUT data from channel
    for (unsigned int reads = 0; reads < SIZE / (BUS_WORDS * N_CHANNELS); reads++) {
        if (!dst.available(1)) {
            std::cerr << "Error: Not enough data in channel to read\n";
            break;
        }

        dst_T dst_pack = dst[reads];

        for (unsigned int words = 0; words < BUS_WORDS; words++) {
            vector_t vector = dst_pack[words];
            for (unsigned int ch = 0; ch < N_CHANNELS; ch++) {
                dut_data.push_back(vector[ch]);
            }
        }
    }

    // Step 3: Compare ref and DUT values
    size_t compare_size = std::min(ref_quantized_data.size(), dut_data.size());
    for (size_t i = 0; i < compare_size; i++) {
        double ref_val = ref_quantized_data[i].to_double();
        double dut_val = dut_data[i].to_double();
        double delta = std::abs(ref_val - dut_val);

        if (delta > threshold) {
            std::cout << "Ref " << ref_val << "  DUT " << dut_val << "     <- MISMATCH" << std::endl;
            ++err_cnt;
        } else if (verbose) {
            std::cout << "Ref " << ref_val << "  DUT " << dut_val << std::endl;
        }
    }

    return err_cnt;
}

} // namespace nnet

#endif
#endif
