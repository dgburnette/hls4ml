#!/bin/bash

CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique"
elif [[ "$OSTYPE" == "linux"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11 -fno-gnu-unique -Wno-pragmas"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++11"
fi
LDFLAGS=

# Pick up AC libraries from Catapult install first
INCFLAGS="-I$MGC_HOME/shared/include -I$MGC_HOME/shared/include/nnet_utils -Ifirmware/ac_types/include -Ifirmware/ac_math/include -Ifirmware/ac_simutils/include -Ifirmware/ac_ipl/include -Ifirmware/nnet_utils -Ifirmware"
PROJECT=myproject
LIB_STAMP=mystamp
INPUT=tb_input_features.dat
OUTPUT=tb_output_predictions.dat
# For proper execution of the shared library from within Python, define the weights dir location
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="\"${BASEDIR}/firmware/weights\""

if [ -d "$MGC_HOME/shared/include/nnet_utils" ]; then
  echo "build_vra.sh: Creating standalone C++ testbench executable with VRA enabled"
  rm -f ${PROJECT}_vra.exe
  ${CC} -std=c++11 -DAC_FIXED_VRA -g ${INCFLAGS} firmware/${PROJECT}.cpp ${PROJECT}_test.cpp -lbfd -o ${PROJECT}_vra.exe
  echo "build_vra.sh: Generating VRA data"
  export AC_FIXED_VRA_OPTS="-f${PROJECT}_raw.csv -d59"
  ./${PROJECT}_vra.exe ./firmware/weights ${INPUT} ${OUTPUT}
fi

