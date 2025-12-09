#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include <ac_channel.h>
#include <iostream>
#include <math.h>

namespace nnet {

namespace product {

/* ---
 * different methods to perform the product of input and weight, depending on the
 * types of each.
 * --- */

class Product {
  public:
    static void limit(unsigned multiplier_limit) {} // Nothing to do here
};

template <class x_T, class w_T> class both_binary : public Product {
  public:
    static x_T product(x_T a, w_T w) {
        // specialisation for 1-bit weights and incoming data
        return a == w;
    }
};

template <class x_T, class w_T> class weight_binary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 1-bit weights, arbitrary data
        if (w == 0)
            return -a;
        else
            return a;
    }
};

template <class x_T, class w_T> class data_binary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-w) {
        // Specialisation for 1-bit data, arbitrary weight
        if (a == 0)
            return -w;
        else
            return w;
    }
};

template <class x_T, class w_T> class weight_ternary : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(-a) {
        // Specialisation for 2-bit weights, arbitrary data
        if (w == 0)
            return 0;
        else if (w == -1)
            return -a;
        else
            return a; // if(w == 1)
    }
};

template <class x_T, class w_T> class mult : public Product {
  public:
    static auto product(x_T a, w_T w) -> decltype(a * w) {
        // 'Normal' product
        return a * w;
    }
    static void limit(unsigned multiplier_limit) {
    }
};

#include <iostream>

template <class x_T, class w_T>
class weight_exponential : public Product {
private:
  using exp_t = decltype(w_T::weight);  // e.g., ac_int<EW, true>
  static constexpr int EXP_W = exp_t::width;
  static constexpr int EMAX  = (1 << (EXP_W - 1)) - 1;

  static constexpr int I_OUT = x_T::i_width + EMAX + 1; // +1 for negation safety
  static constexpr int W_OUT = I_OUT + (x_T::width - x_T::i_width);

public:
  using r_T = ac_fixed<W_OUT, I_OUT, true, x_T::q_mode, x_T::o_mode>;

  static r_T product(const x_T& a, const w_T& w) {

    r_T tmp = static_cast<r_T>(a);

    // Perform shift
    if (w.weight >= 0) {
      tmp <<= w.weight;
    } else {
      tmp >>= -w.weight;
    }

    // Apply sign
    if (w.sign != 1) {
      tmp = -tmp;
    }
    return tmp;
  }
};

} // namespace product

template <class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ac_int<1, false>>::value &&
                                   std::is_same<typename CONFIG_T::weight_t, ac_int<1, false>>::value,
                               ac_int<nnet::ceillog2(CONFIG_T::n_in) + 2, true>>::type
cast(typename CONFIG_T::accum_t x) {
    return (ac_int<nnet::ceillog2(CONFIG_T::n_in) + 2, true>)(x - CONFIG_T::n_in / 2) * 2;
}

template <class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<std::is_same<data_T, ac_int<1, false>>::value &&
                                   !std::is_same<typename CONFIG_T::weight_t, ac_int<1, false>>::value,
                               res_T>::type
cast(typename CONFIG_T::accum_t x) {
    return (res_T)x;
}

template <class data_T, class res_T, typename CONFIG_T>
inline typename std::enable_if<(!std::is_same<data_T, ac_int<1, false>>::value), res_T>::type
cast(typename CONFIG_T::accum_t x) {
    return (res_T)x;
}

} // namespace nnet

#endif
