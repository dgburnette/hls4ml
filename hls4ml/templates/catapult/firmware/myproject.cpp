#include <iostream>

#include "myproject.h"
#include <mc_scverify.h>

#include <ac_shared.h>
#include <ac_sync.h>

#include "parameters.h"

// hls-fpga-machine-learning insert namespace-start

// hls-fpga-machine-learning insert load_scratchpad

#pragma hls_design top
// hls-fpga-machine-learning insert IFSynPragmas
void CCS_BLOCK(myproject)(
    // hls-fpga-machine-learning insert header
) {

    // hls-fpga-machine-learning insert weights

    // hls-fpga-machine-learning insert IO

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        loaded_weights = true;
    }
#endif

    // hls-fpga-machine-learning insert ac_shared

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers
}

// hls-fpga-machine-learning insert namespace-end
