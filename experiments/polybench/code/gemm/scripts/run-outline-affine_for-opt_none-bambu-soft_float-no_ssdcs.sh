#!/bin/bash

# USAGE
#  ./<file-name>.sh <kernel size>

# Definition
#  Calls the scripts to perform soda-opt compilation and bambu synthesis

# Kernel configs
NAME=gemm
KSIZE=$1

# Getting the kernel directory
KERNELDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"/..

# Bambu configs
source ${KERNELDIR}/../../scripts/bambu-config-values.sh

# Load debug flags for bambu
source ${KERNELDIR}/../../scripts/bambu-debug-flags.sh

# Execute
source ${KERNELDIR}/../../scripts/outline-affine_for-opt_none-bambu-soft_float-no_ssdcs.sh