# 原生 PTX GPU 验证链路与新机器交接

更新：2026-10-21。用途：在新机器上继续 HBFSim 基础 GPU 验证，优先支持 H800/SM90 或 A100/SM80 的最小确定性 kernel。**本文是操作与验收指南，不是端到端复现成功证明。**

## 1. 当前检查点与证据边界

交接源码 commit：`58cb0d7674720c1ba5990da0e8d1455caa5155a5`。原机器项目目录为 `/home/data/yjl/HBFSim`，`/home/yjl/HBFSim` 为兼容入口。该 commit 未包含原机器 ignored `artifacts/` 中的隔离修复；本文的[文本附件](native-ptx-validation-assets/)保存了继续工作所需的源码和模板，生产源码未被替换。

| 项目 | 已有结果 | 不能据此推断的结果 |
|---|---|---|
| H800 原生 CUDA microbenchmark | 真实 GPU 执行、CPU checksum 校验通过；64KiB、16次循环、seed=1 | 插桩后代码执行成功 |
| SHA256 兼容插件 | 已构建；10组边界向量和空输出API检查通过；同一输入的完整插件JSON/PTX与原版逐字节一致；实机已越过原崩溃点 | 完整 Timing 链路通过 |
| 独立 bpftime agent | trampoline 声明由 PTX 8.1 改为8.7，已构建 | 修复后的 agent 已完成实机验收 |
| SM90 合并 PTX | 当前插件离线重写24条访存，合并后通过 SM90 ptxas，反汇编 ELF 为SM90 | 本轮 GPU 实际加载了该 cubin |
| 兼容插件定向回归 | PTX plugin、device helper PTX、helper ABI：3 passed | 历史完整回归已经全绿 |
| 历史 CPU/离线完整回归 | 44 passed、14 failed、0 skipped；原始失败保留 | 真实 GPU 失败数量或成功数量 |
| SM80 helper 与 kernel | compute_80 编译、sm_80 汇编共4项通过 | A100 实机或完整 SM80 插桩插件已验证 |
| 最新 H800 启动 | 外部PID占用、87°C、显存约73.6GiB，启动门槛拒绝；未启动 kernel | kernel 功能失败 |

原始日志目录：`artifacts/h800-gpu-baseline-v1/`、`artifacts/h800-gpu-baseline-v2/`。本轮阶段报告为后者的 `REPORT.md`。原机器的绝对路径、GPU UUID、空闲状态和工具链不能直接当作新机器元信息。

证据身份约定：

- **USER_CONFIRMED**：用户要求原生 PTX、禁用 SASS/Futures、真实 GPU 验收、保留原始失败和结果。新机器执行前还须确认其具体设备和小规模执行范围。
- **DOC_DERIVED**：基础 helper 源码要求 CC>=7.0；CMake允许单一数字架构；workload和内嵌helper必须为相同baseline target。详见[架构兼容审计](cuda-architecture-compatibility.md)。
- **MEASURED**：上表明确标出的实际 GPU、构建、离线检查结果。
- **SCENARIO_ASSUMPTION**：附件 profile 的1000/100000ns等参数仅用于软件功能验证，不是实物HBF延迟、功耗或论文性能标定。
- **INFERRED**：SM80离线通过说明A100值得继续验证，不等于主机/GPU共享协议已经通过。

## 2. 本阶段链路与范围

```text
未修改的原生目标 PTX
  → bpftime 识别 hbf_access_kernel 并调用 HBFSim PTX pass
  → 注入匹配架构的 __hbfsim_resolve 等 helper
  → 合并 trampoline，统一模块声明，编译为目标 SM cubin
  → exact-load provenance / launch gate 验证和控制区绑定
  → kernel launch 选择修改后的函数
  → GPU helper 判断地址范围
       未注册：原生 HBM 访问
       已注册：fast 或 reference/MQSim 请求、等待、原访存执行
  → CUDA 同步、checksum和完整数据比较、GPU访问计数、coverage验收
```

运行入口保持为 `scripts/run_with_bpftime.sh`。关键源码包括 `src/ptxpass_hbf/plugin.cpp`、`transform.cpp`、`src/cuda_runtime/device/hbf_device.cu`、`context.cpp`、`launch_gate.cpp`，以及 bpftime 构建副本中的 `nv_attach_impl.cpp`、`nv_attach_fatbin_record.cpp` 和 `nv_attach_impl_frida_setup.cpp`。

只验证普通global load/store、Timing-only、1个block/1个thread、64KiB、16次循环、seed=1。禁用SASS恢复、ULDC恢复开发、cuBLAS适配、完整vLLM/LLM、长性能实验和SM120专属Timing Futures。没有专门的“禁用SASS”顶层CMake选项：不要编造选项；本指南完全不调用lifter或相关自动恢复脚本。

默认同步helper使用 **issue-point waiting**，不报告dependency-point等待或延迟隐藏结论。GPU wall time不要求与注入延迟线性一致。

## 3. 新机器先检查什么

先读该机器的 `AGENTS.md`、交接记录、`README.zh-CN.md`、[PTX插桩](../skills/02-ptx-instrumentation.md)、[CUDA内存语义](../skills/03-cuda-memory-semantics.md)、`include/hbfsim/api.h`及实际脚本。本文不覆盖仓库规则或执行授权。

每次使用全新的构建和运行目录。以下环境变量在仓库根目录设置；激活新机已有的Python环境，并按实际安装调整CUDA、LLVM和编译器路径：

```bash
set -euo pipefail
export ROOT="$PWD"
export CUDA_ROOT=/usr/local/cuda-12.8
export LLVM_ROOT=/usr/lib/llvm-15
export SM=80                         # A100=80；H800=90
export RUN_ID=a100-native-ptx-v1      # 新的、尚不存在的实验ID
export WORK="$ROOT/artifacts/$RUN_ID"
export BUILD="$ROOT/build-$RUN_ID"
export AGENT="$WORK/build-bpftime-ptx87"
export ASSETS="$ROOT/docs/reference/native-ptx-validation-assets"
export PATH="$LLVM_ROOT/bin:$CUDA_ROOT/bin:$PATH"
test ! -e "$WORK"
test ! -e "$BUILD"
mkdir -p "$WORK/logs"
```

不要在旧结果目录重新执行初始化。保存以下命令的原始输出和退出码：

```bash
git rev-parse HEAD
git status --short
git submodule status --recursive
nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,temperature.gpu,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvcc --version
ptxas --version
cuobjdump --version
nvdisasm --version
cmake --version
ninja --version
g++-12 --version
clang-15 --version
llvm-strip --version
pkg-config --modversion libbpf libelf openssl
python -c 'import sys; print(sys.executable); print(sys.version)'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
bash scripts/build_patched_bpftime.sh --check
```

原环境为CUDA12.8.93、驱动580.82.07、GCC12.3、LLVM15.0.7、Boost1.74、OpenSSL3.0.2、CMake3.31.6。新机不必完全相同，但要记录差异并验证驱动/Toolkit兼容性。基础微基准并不调用PyTorch或vLLM；不要把它们的import成功当作HBFSim通过。

启动前保存 `PREFLIGHT.md` 与机器可读manifest：设备UUID、源码/submodule commit、实际so/PTX及profile哈希、命令、注册范围、重复数、预算、停止条件、预期输出。明确记录执行授权来源，不伪造用户签名或用代理生成的manifest当作批准。

旧有小规模计划：每点一次、串行；每点90s、请求timeout5s；启动时无其他目标GPU计算进程、温度<=70°C、显存<=512MiB、利用率<=10%；运行中温度>80°C、目标GPU总显存>4GiB、进程树RSS>8GiB或主机余量不足时停止本任务。新机器要按其资源重新确认合理的有限预算；不能因预算不合适静默删掉监控。不得终止其他用户的进程。

## 4. 必须带走的隔离修复

[附件目录](native-ptx-validation-assets/)是文本源码，不含旧机器二进制或原始实验数据：

| 附件 | 用途 |
|---|---|
| `sha256_namespace_compat.cpp` | 独立插件的hidden `--wrap=SHA256`兼容实现 |
| `agent-ptx87.cmake`、`agent-trampoline-version.diff` | 只在bpftime复制源码中修正版本声明；核对原/目标源码哈希 |
| `compat/boost/process/v1.hpp` | **仅Boost1.74**构建shim；其他版本不得直接套用 |
| `range_microbench.cpp`、`range-microbench.diff` | 复用原kernel和CPU oracle，增加none/full/partial、exact binder、已有访问计数和全数据回读 |
| `hbfsimd_observed.cpp`、`daemon-observer.diff` | 独立daemon启用既有MQSim observations，记录实际完成事件；不改协议或调度 |
| `profiles/small.json`、`large.json` | 相同几何，read/program分别1000ns和100000ns |
| `dlmopen_probe.cpp`、`sha_probe.cpp` | 兼容插件隔离加载和哈希API的固定检查源码 |
| `gpu_stage.py.example`、`run_batch.py.example` | 旧安全启动器/证据验证器的原样模板；**须按第9节移植，不能原样执行** |
| `provenance.json` | 原样复制附件的来源路径和SHA256；后加的便携CMake hook单独列为本次文档附件 |

旧原始日志可单独复制存档；不必迁移整个构建树。不要复用旧CMakeCache、旧SM90 `.so`、CUDA cache或绝对路径链接。A100必须生成SM80 helper、嵌入它的core/plugin及SM80 workload PTX。

SHA兼容修复的证据是：原运行时在`dlmopen(LM_ID_NEWLM)`加载插件后的OpenSSL默认provider/context锁路径崩溃。兼容实现仍使用同一OpenSSL SHA-256算法的Init/Update/Final，避免该默认provider入口；不据此推广为所有OpenSSL命名空间问题均已解决。

## 5. 构建主项目与目标 helper

```bash
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -UNDEBUG' \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES="$SM" \
  -DHBFSIM_ENABLE_CUDA=ON \
  -DHBFSIM_ENABLE_MQSIM=ON \
  -DHBFSIM_ENABLE_TIMING_FUTURES=OFF \
  -DHBFSIM_ENABLE_LLM_TESTS=OFF \
  -DHBFSIM_GATE_CUDART_VERSION_MAP="$ROOT/cmake/cudart-versions.map" \
  > "$WORK/logs/configure.log" 2>&1

cmake --build "$BUILD" --parallel 2 > "$WORK/logs/build.log" 2>&1
rg 'CMAKE_CUDA_ARCHITECTURES|HBFSIM_ENABLE_(CUDA|MQSIM|TIMING_FUTURES)' "$BUILD/CMakeCache.txt"
rg '^\s*\.(version|target|address_size)' "$BUILD/generated/hbf_device.ptx" "$BUILD/benchmarks/cuda/hbf_microbench_kernel.ptx"
```

这里的`-UNDEBUG`用于保留原CPU测试assert，不改预期。若构建失败，先记录最早实质性错误和命令退出码；不要只记录最终Make Error 2。

## 6. 独立构建修复后的 bpftime

保留pinned bpftime commit `ec26daecc8e787fb80fd95dd596a576404a5e36e`。现有项目脚本先应用`patches/bpftime/0001-exact-module-load-provenance.patch`，不要手工伪造构建stamp。版本修复通过附件hook只作用于新构建目录的`_deps/bpftime-hbfsim-src`。

以下配置对应已审查的Boost1.74环境。先检查`boost/version.hpp`；其他Boost版本应核查其Process布局，并去掉不适用的shim，不要无依据升级整个工具链。

```bash
export HBFSIM_CUDA_ROOT="$CUDA_ROOT"
export HBFSIM_BUILD_JOBS=2
test ! -e "$AGENT"
bash scripts/build_patched_bpftime.sh "$AGENT" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CXX_FLAGS="-I$ASSETS/compat" \
  -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" \
  -DCMAKE_PROJECT_bpftime_INCLUDE_BEFORE="$ASSETS/agent-ptx87.cmake" \
  > "$WORK/logs/build-agent.log" 2>&1

cat "$AGENT/hbfsim-bpftime.provenance"
sha256sum "$AGENT/_deps/bpftime-hbfsim-src/attach/nv_attach_impl/trampoline_ptx.h" \
  "$AGENT/runtime/agent/libbpftime-agent.so" \
  "$AGENT/runtime/syscall-server/libbpftime-syscall-server.so" \
  > "$WORK/agent-extra-provenance.sha256"
```

额外修复后的trampoline源码SHA256应为`2e8763802dbe7ae5261cdb37686081e551b298203a71a905916b3582c73cf611`。项目原stamp只证明base patch，**不覆盖额外版本修复**；上面的额外记录与构建日志都要保留。

常见构建问题：

- `llvm-strip`：使用匹配LLVM15工具链的真实路径；修改本次进程PATH即可。
- Frida：固定16.1.2；可将已核验的两个linux-x86_64 devkit放入单独cache，并设置`HBFSIM_FRIDA_CACHE`后再配置；hook不关闭原CMake校验。SHA256见`third_party/bpftime/cmake/frida.cmake`。该便携hook的cache分支只适用x86_64。
- 构建中断：不要反复对同一目录运行复制源码的顶层脚本，它可能与旧ExternalProject stamp冲突。保留失败目录，使用新的目录；仅继续既有ninja构建前也要先确认源码/patch身份。
- 出现真实新错误：最多进行有根因证据的有限修复；不禁用CUDA attach、exact provenance或coverage功能换取构建通过。

## 7. 独立兼容插件与最小验证程序

编译到新文件，保持原`$BUILD/libptxpass_hbf.so`不变：

```bash
g++-12 -std=c++20 -O3 -UNDEBUG -fPIC -shared \
  -Wl,--wrap=SHA256 \
  -I"$ROOT/include" -I"$ROOT/src/ptxpass_hbf" \
  -I"$ROOT/third_party/bpftime/third_party" \
  "$ROOT/src/ptxpass_hbf/plugin.cpp" "$ASSETS/sha256_namespace_compat.cpp" \
  "$BUILD/libhbfsim_core.a" -lcrypto -pthread \
  -o "$WORK/libptxpass_hbf_sha_compat.so" \
  > "$WORK/logs/build-compat-plugin.log" 2>&1

g++-12 -std=c++20 -O2 -UNDEBUG -Wl,--export-dynamic \
  -I"$ROOT/include" -I"$CUDA_ROOT/targets/x86_64-linux/include" \
  "-DHBFSIM_MICROBENCH_PTX_PATH=\"$BUILD/benchmarks/cuda/hbf_microbench_kernel.ptx\"" \
  "$ASSETS/range_microbench.cpp" -o "$WORK/range_microbench" \
  "$BUILD/libhbfsim_core.a" \
  "$CUDA_ROOT/targets/x86_64-linux/lib/libcudart.so" \
  "$CUDA_ROOT/targets/x86_64-linux/lib/stubs/libcuda.so" \
  -ldl -lcrypto "$BUILD/libmqsim_hbf.a" -pthread -lrt \
  -Wl,-rpath,"$CUDA_ROOT/targets/x86_64-linux/lib" \
  > "$WORK/logs/build-fixture.log" 2>&1

g++-12 -std=c++20 -O2 -UNDEBUG -DHBFSIM_ENABLE_MQSIM_RUNTIME=1 \
  -I"$ROOT/include" -I"$ROOT/src/host_service" \
  "$ASSETS/hbfsimd_observed.cpp" -o "$WORK/hbfsimd_observed" \
  "$BUILD/libhbfsim_core.a" \
  "$CUDA_ROOT/targets/x86_64-linux/lib/libcudart.so" \
  "$CUDA_ROOT/targets/x86_64-linux/lib/stubs/libcuda.so" \
  -ldl -lcrypto "$BUILD/libmqsim_hbf.a" -pthread -lrt \
  -Wl,-rpath,"$CUDA_ROOT/targets/x86_64-linux/lib" \
  > "$WORK/logs/build-observer.log" 2>&1
```

`stubs/libcuda.so`只用于链接；**运行时不得将stubs目录放入LD_LIBRARY_PATH**。实际`libcuda.so.1`应解析到新机驱动，不是测试fake driver。附件使用原kernel源码生成的PTX，不是手工预插桩PTX；独立驱动增加观察接口，没有重新实现PTX pass。

为原wrapper建立独立视图，明确选择兼容插件：

```bash
export VIEW="$WORK/build-view"
mkdir "$VIEW"
ln -s "$BUILD/libhbfsim_launch_gate.so" "$VIEW/libhbfsim_launch_gate.so"
ln -s "$WORK/libptxpass_hbf_sha_compat.so" "$VIEW/libptxpass_hbf.so"
ln -s "$BUILD/hbfsim_bpftime_attach_loader" "$VIEW/hbfsim_bpftime_attach_loader"
ln -s "$BUILD/microbench_probe.bpf.o" "$VIEW/microbench_probe.bpf.o"
readlink -f "$VIEW/libptxpass_hbf.so"
sha256sum "$VIEW/libptxpass_hbf.so" "$VIEW/libhbfsim_launch_gate.so"
```

必须选`microbench_probe.bpf.o`：wrapper默认`coverage_probe.bpf.o`的kernel名不匹配`hbf_access_kernel`。

## 8. GPU前的固定检查

```bash
CUDA_VISIBLE_DEVICES='' python tests/integration/test_ptxpass_plugin.py \
  "$WORK/libptxpass_hbf_sha_compat.so" "$ROOT/src/ptxpass_hbf/plugin.cpp" \
  "$CUDA_ROOT/bin/ptxas" "sm_$SM" > "$WORK/logs/test-plugin.log" 2>&1

CUDA_VISIBLE_DEVICES='' python tests/integration/test_device_helper_ptx.py \
  "$WORK/libptxpass_hbf_sha_compat.so" "$BUILD/generated/hbf_device.ptx" \
  "$CUDA_ROOT/bin/ptxas" "sm_$SM" > "$WORK/logs/test-helper.log" 2>&1

CUDA_VISIBLE_DEVICES='' "$BUILD/hbfsim_device_helper_abi_tests" \
  > "$WORK/logs/test-abi.log" 2>&1
```

另用附件`sha_probe.cpp`构建带相同wrap的哈希probe，对0/3/63/64/65/127/128/129/4096/1048576字节及null-output API与Python hashlib比较。`dlmopen_probe.cpp`以`-ldl`编译，输入沿用plugin的`process_input` JSON协议；分别对原/兼容插件运行并比较完整响应JSON与PTX。必须使用**新构建的目标架构输入**，不要拿旧SM90输入验证SM80插件。

PTX检查应覆盖：单一版本声明、单一target和64位地址；输入kernel与helper为同一numeric SM；保留全部指令，由真实`ptxas -arch=sm_$SM`检查；resolver参数、返回结构、control及generation符号相符。参数`ld.param`并非HBF global访存：原始coverage统计可能记录为不改写，但应以manifest中HBF相关unsupported分类及实际gate决策验收，不能删指令或忽略unsafe覆盖。

版本8.7是本次CUDA12.8生成物的ISA版本，不是SM90专属功能。只提升trampoline声明并不足以证明兼容；还须完成上述检查、实际编译和GPU协议验证。不要将旧目标PTX简单改名为新目标来代替重新编译helper。

## 9. 移植安全启动器，然后运行有界用例

附件`*.py.example`是原机器模板，保留其原始哈希用于追溯；它们尚不是新机器可直接运行的脚本。复制到本次独立`WORK`后，至少审查并更新以下消费者，再生成新的manifest：

1. `ROOT/BASE/OUT/BUILD/AGENT`：真实仓库、本次输出、build-view和agent目录；不能继续指向v1旧视图。
2. UUID、GPU名称、SM、驱动和所有元信息；`BPFTIME_SM_ARCH`与native PTX目录必须匹配目标。
3. Python、CUDA/LLVM、动态库、磁盘挂载和安全余量；旧`/home/yjl`空间检查不能照搬。
4. 多GPU机器的所有`nvidia-smi`查询加目标`-i UUID`，只解析目标GPU状态及其计算进程；原模板按单卡CSV解析，不能用于多卡原样输出。
5. `native-ptx`目录只放新构建的**未插桩**kernel PTX；fixture、daemon和profile链接均指向本轮产物。
6. 原模板最后的`capacity-live`指向旧隔离输入修复测试；先暂缓这个独立补充用例并明确记为NOT RUN，完成下面Timing主线。不要把旧SM90二进制带入SM80，也不要因它缺失取消主线gate。
7. 验证器的import路径、输出路径和case列表与启动器一致。停止条件、strict policy和实际计数断言保留；任何更改都写入changes和新manifest。

先执行`--prepare-only`并人工核对命令、解析后的so路径及哈希；不存在有效manifest时不得启动。不要重跑旧目录：新attempt用新输出目录和新manifest，失败记录保留。不要后台无限等待GPU，也不要逐秒由代理轮询；脚本内部资源监控必须保留。

### 用例顺序

| 顺序 | 程序/模式 | 注册策略 | profile | 目的 |
|---|---|---|---|---|
| 1 | 原仓库microbench / baseline | 无模型 | small | 原始CUDA环境及oracle |
| 2 | 独立fixture / baseline | none | small | 保存完整64KiB数据及CUDA event时间 |
| 3 | fixture / fast | none | small | 修改后代码真实运行，但零HBF请求 |
| 4 | fixture / fast | full | small | 注册命中、实际建模 |
| 5 | fixture / fast | partial | small | 同一kernel有HBF/HBM两条路径 |
| 6 | fixture / fast | full | large | modeled latency随profile变化 |
| 7 | 原仓库microbench / fast | full | small | 验证仓库原入口；不以它的静态报告字段代替计数 |
| 8–11 | fixture / reference | none/full/partial/full | small/small/small/large | 仅在fast通过后验证MQSim |
| 12 | compute-sanitizer / baseline fixture | none | small | 小型memcheck补充；不能代替插桩正确性 |

每点1次，数据65536字节、16迭代、seed1、mixed_rw；partial只注册前32768字节，full注册65536字节。offset数组与输出地址始终未注册，因此full也应有native out-of-range访问。

安全启动器应执行的原始命令形式如下，**仅是启动器内部命令模板，不代替资源监控与启动门槛**：

```bash
# CASE_DIR必须是本次新建的独占目录，事先建立其report和tmp目录。
# baseline直接运行；fast/reference由同一安全启动器附加下面的环境并调用wrapper。
"$WORK/range_microbench" --pattern mixed_rw --mode baseline \
  --bytes 65536 --iterations 16 --seed 1 --registration none \
  --profile "$ASSETS/profiles/small.json" \
  --report-dir "$CASE_DIR/report" --output "$CASE_DIR/result.json"

HBFSIM_BUILD_DIR="$VIEW" \
HBFSIM_BPFTIME_BUILD_DIR="$AGENT" \
HBFSIM_BPFTIME_PROBE="$VIEW/microbench_probe.bpf.o" \
HBFSIM_CUDA_ROOT="$CUDA_ROOT" \
HBFSIM_DAEMON_PATH="$WORK/hbfsimd_observed" \
HBFSIM_SOFTWARE_CHECK_MQSIM_TRACE="$CASE_DIR/mqsim-events.jsonl" \
HBFSIM_INSTRUMENTATION_POLICY=strict \
HBFSIM_COVERAGE_PATH="$CASE_DIR/coverage.jsonl" \
HBFSIM_PASS_MANIFEST_PATH="$CASE_DIR/pass-manifests.jsonl" \
BPFTIME_SM_ARCH="sm_$SM" \
BPFTIME_CUDA_LATE_PTX_DIR="$WORK/native-ptx" \
BPFTIME_LOG_OUTPUT="$CASE_DIR/bpftime.log" \
SPDLOG_LEVEL=debug \
bash "$ROOT/scripts/run_with_bpftime.sh" -- \
  "$WORK/range_microbench" --pattern mixed_rw --mode fast \
  --bytes 65536 --iterations 16 --seed 1 --registration full \
  --profile "$ASSETS/profiles/small.json" \
  --report-dir "$CASE_DIR/report" --output "$CASE_DIR/result.json"
```

原仓库用例将程序换为`$BUILD/benchmarks/cuda/hbf_microbench`，并去掉`--registration`参数，其余数据参数一致。外层启动器设置目标`CUDA_VISIBLE_DEVICES=UUID`、独立TMPDIR/CUDA cache、OMP/BLAS=1，清理继承的HBFSIM/BPFTIME和旧LD_PRELOAD，记录有效环境及退出码。运行时LD_LIBRARY_PATH只包含正确Toolkit运行库及必要真实依赖。

## 10. 必须采集的实际执行证据

每点至少保存：完整命令/环境/输入hash、GPU前置状态、stdout/stderr、bpftime debug日志、pass-manifests、coverage、result JSON、access/binding sidecar、完整数据文件、CUDA event时间、退出码及停止原因。

开启debug后，agent会在`/tmp/bpftime-fatbin-work.*`写出patched PTX。从**本点日志给出的准确路径**复制到独占结果目录，不扫描搬走其他任务的tmp文件，不删除原始文件。注意：这个dump位于最后`rewrite_ptx_target()`之前，可能仍显示trampoline的旧默认target；**不能把它直接称为最终编译输入**。结合`compile_ptxs`源码、实际gpu-name编译选项和日志核对；若缺少最终输入或实际加载证据，在隔离agent中增加默认关闭的日志/转储，记录diff、哈希并重建后重试。

验收条件必须同时满足：

- pass manifest对应`hbf_access_kernel`、目标SM、`instrumented=true`、重写数>0、HBF相关unsupported=0；模块ID对应本次原生PTX哈希。
- strict coverage有实际launch准入记录，`allowed=true`且非opaque/unmodeled；无绕过gate。
- fixture的exact binder状态0，access snapshot为`COMPLETE`、module_count>0、disable_complete；不能只看binder sidecar宣称已执行。
- GPU计数`supported_accesses>0`，且等于`in_range_accesses + native_out_of_range_accesses`；所有失败/未分类/overflow计数为0。
- `modeled_admitted_accesses == service_completed_accesses`；已注册用例二者及`service_requests`非零。
- none注册：in-range、modeled、service requests均0，但native访问非零；partial的建模访问数严格介于none/full之间，native数相反。
- 每点checksum等于CPU oracle；完整`result.json.data.bin`与baseline逐字节一致，长度65536；CUDA同步成功。

原microbench/fixture报告中的`coverage.unsafe_launches=0`是固定字段，**不能用作实际覆盖证据**。必须读真实coverage JSONL和GPU访问快照。exact模块身份、控制别名和修改后函数选择共同约束执行路径；仅PTX pass返回成功、生成cubin或注册API成功均不构成GPU命中证明。

## 11. Timing数值如何解释

fast用`result.json`中的`requests.fast`和`timing.modeled_ns`，以及GPU访问计数交叉验证。full/large的modeled latency应大于full/small；none应为0。

reference用`requests.reference`与submitted/completed平衡，并读取独立daemon记录的MQSim完成事件`kind=2`。每个事件验证`modeled_completion_ns > arrival_ns`，分别保存事件数和`sum(modeled_completion_ns-arrival_ns)`。日志保留`time_ns`，不把callback时间、模拟完成时间和GPU wall time混为一谈。`fast_modeled_ns=0`不能解释为reference没有模型延迟。

两份profile容量16GiB、页16KiB、HBM-role cache64MiB、32channel×8die×4plane。它们用于软件检查；不要将新GPU的HBM容量填成HBF器件容量。loader拒绝零read/program latency和零time_scale，所以“不注入”对照使用baseline/none，不删除profile校验构造registered-zero测试。

分别记录modeled work、CUDA event时间、主机wall time。reference观察日志存在主机开销；没有独立测量时，simulator overhead写NOT MEASURED，不能用wall减modeled time臆造。不要将profile差异解释为实物A100/H800之间的性能或HBF校准差异。

## 12. 新错误的定位顺序

| 最早直接错误 | 首先检查 | 不允许的捷径 |
|---|---|---|
| 插件加载/SHA锁崩溃 | 实际加载路径、兼容插件hash、dlmopen调用栈和namespace映射 | 宣称是CUDA版本问题后盲目全局升级 |
| used_bytes_mask版本错误 | 合并首声明、真实编译输入、trampoline额外patch和helper生成版本 | 删指令或不验证便只改版本数字 |
| helper target mismatch | core/plugin内嵌架构与workload是否同一SM | 把SM90 PTX头改为SM80来伪装重编译 |
| 无pass/coverage或name不匹配 | microbench probe、插件配置、原生PTX目录、exact binder | 关strict、预写成功manifest、按kernel名猜模块 |
| gate拒绝或零命中 | 完整准入原因、真实参数ABI、注册范围、访问快照及生命周期 | 把注册成功当作命中 |
| 同步超时/非法访存 | CUDA最早错误、控制区绑定、producer模式、请求/完成和精确调用栈 | 延长到无界等待或绕过错误检查 |
| reference建模不符 | MQSim实际事件、profile消费者、请求完成状态 | 用fast字段或人工结果代替reference证据 |

崩溃先收集精确调用栈和地址映射；对同一问题做有限、可解释的重试，失败点新建目录。GDB自身退出0不代表被调试程序通过。`HostNativeAtomicSupported`、mapped-host注册和VMM支持都必须实测；源码有GPU-exclusive producer分支，但它不等于新平台共享协议验证。

## 13. 回归、报告和完成判据

完整CPU/离线回归串行运行并保存XML/日志，GPU独立运行，不混算：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ctest --test-dir "$BUILD" --output-on-failure --parallel 1 \
  --output-junit "$WORK/cpu-tests.xml" > "$WORK/logs/cpu-tests.log" 2>&1
```

预期可能复现的14项历史失败：2项测试输入硬编码SM120；9项public CUDA lifecycle及1项vLLM extension的fake driver缺新属性查询；1项capacity构造事件旧预期7、实际8；1项launch_gate_symbols整文件`expected_`子串规则。全部保留为FAILED，分别判断是否影响真实GPU主线；不要修改预期、删除测试或用定向3项通过覆盖完整回归结果。

独立VMM live原测试还有tiny_profile缺8个必填per-level latency的输入问题；v1只在副本中补齐并保留原检查。Timing主线完成后，再审查这个fixture适用性并执行补充live测试。原CPU/离线SASS测试不属于本轮GPU证明。

最终目录至少有`environment.md`、manifest/preflight、build日志、每点原始运行目录、`coverage.json`汇总、`timing.json`汇总、`tests.log`/XML、`changes.md`和`REPORT.md`。汇总文件引用原始证据，不能改写raw；分别列出Passed/Failed/Skipped/Not run。无真实GPU证据的字段写NOT RUN/NOT MEASURED。

只有构建、自动插桩、修改后SM代码实际加载执行、注册命中、非零建模延迟、未注册原生路径、profile差异、输出一致和严格覆盖均有实机证据，且功能性失败均得到解释，才可宣布基础GPU复现成功。A100完成后只能报告A100/SM80的验证范围，不能代替H800/SM90或原论文SM120结果。

下一阶段接入vLLM的最小工作集：先固定其Torch/vLLM/算子版本和目标架构；核查实际算子原生PTX与coverage；完成真实权重地址注册及exact模块身份；以小模型/小输入单独批准验证。预编译库只有cubin而缺适用PTX时属于新的阻塞，不能用本微基准成功宣称已经解决。本阶段不继续开发SASS恢复器。
