#!/usr/bin/env bash
set -euo pipefail
root=/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1
source "$root/activate.sh"
# Software construction only: no GPU tests or workload launches.
[ "$(df --output=avail -B1 "$root" | tail -1)" -ge 32212254720 ]
cmake -S /home/yjl/HBFSim -B "$root/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHBFSIM_ENABLE_CUDA=ON -DHBFSIM_ENABLE_MQSIM=ON \
  -DHBFSIM_ENABLE_TIMING_FUTURES=OFF -DHBFSIM_ENABLE_LLM_TESTS=OFF \
  -DHBFSIM_GATE_CUDART_VERSION_MAP=/home/yjl/HBFSim/cmake/cudart-versions.map \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DHBFSIM_PYTHON3_EXECUTABLE="$root/env/bin/python" \
  > "$root/logs/configure.log" 2>&1
nice -n 10 cmake --build "$root/build" --parallel 2 --target \
  hbfsim hbfsimd hbfsim_launch_gate ptxpass_hbf ptxpass_hbf_plugin \
  hbfsim_cpu_tests hbfsim_profile_tests hbfsim_protocol_tests \
  hbfsim_timing_binding_tests module_identity_test ptx_transform_test \
  > "$root/logs/build.log" 2>&1
ctest --test-dir "$root/build" --output-on-failure \
  -R '^(build_smoke|profile|protocol|timing_binding|module_identity|ptx_transform)$' \
  > "$root/logs/ctest.log" 2>&1
printf '%s\n' DONE > "$root/BUILD_DONE"
