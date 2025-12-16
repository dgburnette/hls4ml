#!/bin/bash
set -e
CC=g++
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    CFLAGS="-O3 -fPIC -std=c++17 -fno-gnu-unique"
elif [[ "$OSTYPE" == "linux"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++17 -fno-gnu-unique -Wno-pragmas"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-O3 -fPIC -std=c++17"
fi
LDFLAGS=

# Pick up AC libraries from Catapult install first
INCFLAGS="-I$MGC_HOME/shared/include -I$MGC_HOME/shared/include/nnet_utils -Ifirmware/ac_types/include -Ifirmware/ac_math/include -Ifirmware/ac_simutils/include -Ifirmware/ac_ipl/include -Ifirmware/nnet_utils -Ifirmware"
PROJECT=myproject
LIB_STAMP=mystamp
# For proper execution of the shared library from within Python, define the weights dir location
BASEDIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHTS_DIR="\"${BASEDIR}/firmware/weights\""


echo "build_lib.sh: Creating shared library for execution from Python"
${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c firmware/${PROJECT}.cpp -o ${PROJECT}.o
${CC} ${CFLAGS} ${INCFLAGS} -D WEIGHTS_DIR="${WEIGHTS_DIR}" -c ${PROJECT}_bridge.cpp -o ${PROJECT}_bridge.o
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o ${PROJECT}_bridge.o -o firmware/${PROJECT}-${LIB_STAMP}.so
rm -f *.o

if [ -d "$MGC_HOME/shared/include/nnet_utils" ]; then
  echo "build_lib.sh: Creating standalone C++ testbench executable"
  rm -f ${PROJECT}.exe
  ${CC} -std=c++17 -g ${INCFLAGS} firmware/${PROJECT}.cpp ${PROJECT}_test.cpp -o ${PROJECT}.exe
  echo ""
  echo "To run the C++ testbench standalone:"
  echo "  ${PROJECT}.exe ./firmware/weights ./tb_data/tb_input_features.dat ./tb_data/tb_output_predictions.dat"
fi
