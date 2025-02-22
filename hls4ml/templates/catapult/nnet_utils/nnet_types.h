#ifndef NNET_TYPES_H_
#define NNET_TYPES_H_

#include <assert.h>
#include <cstddef>
#include <cstdio>

namespace nnet {

// Fixed-size array
template <typename T, unsigned N> struct array {
    typedef T value_type;
    static const unsigned size = N;

    T data[N];

    T &operator[](size_t pos) { return data[pos]; }

    const T &operator[](size_t pos) const { return data[pos]; }

    array &operator=(const array &other) {
        if (&other == this)
            return *this;

        assert(N == other.size && "Array sizes must match.");

        #pragma hls_unroll
        for (unsigned i = 0; i < N; i++) {
            data[i] = other[i];
        }
        return *this;
    }
};

// Generic lookup-table implementation, for use in approximations of math functions
template <typename T, unsigned N, T (*func)(T)> class lookup_table {
  public:
    lookup_table(T from, T to) : range_start(from), range_end(to), base_div(ac_int<16, false>(N) / T(to - from)) {
        T step = (range_end - range_start) / ac_int<16, false>(N);
        for (size_t i = 0; i < N; i++) {
            T num = range_start + ac_int<16, false>(i) * step;
            T sample = func(num);
            samples[i] = sample;
        }
    }

    T operator()(T n) const {
        int index = (n - range_start) * base_div;
        if (index < 0)
            index = 0;
        else if (index > N - 1)
            index = N - 1;
        return samples[index];
    }

  private:
    T samples[N];
    const T range_start, range_end;
    ac_fixed<20, 16, true> base_div;
};

} // namespace nnet

#endif
