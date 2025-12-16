#ifndef NNET_BIAS_H_
#define NNET_BIAS_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include <ac_channel.h>
#include <iostream>
#include <math.h>

namespace nnet {

// PO2 bias: struct with .sign and .weight
template <typename AccT, typename BiasT>
inline auto bias_to_accum(const BiasT& b)
    -> decltype((void)b.sign, (void)b.weight, AccT())  // SFINAE: only if fields exist
{
    AccT mag = AccT(1);
    int e = b.weight.to_int();
    if (e >= 0) mag <<=  e;
    else        mag >>= -e;
    return (b.sign == 1) ? mag : AccT(-mag);
}

// Direct numeric bias (convertible to AccT)
template <typename AccT, typename T,
          typename = typename std::enable_if<std::is_convertible<T, AccT>::value>::type>
inline AccT bias_to_accum(const T& b) {
    return static_cast<AccT>(b);
}

} // namespace nnet

#endif