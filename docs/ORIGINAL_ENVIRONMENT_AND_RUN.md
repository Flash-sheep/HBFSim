# original 分支：环境配置与运行方法

本文对应原版源码基线 `874a20f289b7bd587fa4048a788e12df2618444b`。
`original` 在该基线上补充使用文档；`sm_80支持` 保存另行开发的 CUDA11/sm80
隔离适配代码、补丁和验证记录。本文命令只使用原版入口，不依赖另一分支的脚本。

## 1. 环境需求

| 路径 | 必需条件 |
| --- | --- |
| CPU/MQSim | 原生 Linux、Git、CMake ≥3.25、Ninja、支持 C++20 的编译器、OpenSSL 开发头文件/库、Python3 |
| CUDA 构建 | 以上条件，加 CUDA Toolkit ≥12.8；主机编译器须同时兼容选用的 Toolkit 和 C++20 |
| GPU 实际运行 | 可用 NVIDIA GPU、匹配的驱动/运行库，以及目标架构兼容的 kernel PTX；构建成功不等于运行验证成功 |
| bpftime CUDA attach | 固定子模块与项目补丁；另需 LLVM/Clang、Boost、libbpf、libelf、zlib/ncurses 等构建依赖，以固定源码的 CMake 检测为准 |
| LLM workload | 对应适配器要求的 Python 包、模型权重和原始 workload；这些不随仓库分发 |

不要求特定名称的 Conda 环境。可使用独立 Conda/venv 安装 Python 构建工具，避免
修改全局环境；OpenSSL 等系统开发库仍须能够被 CMake 找到。例如：

```bash
python3 -m venv ../hbfsim-original-env
source ../hbfsim-original-env/bin/activate
python -m pip install 'cmake>=3.25' ninja
cmake --version
ninja --version
c++ --version
```

这只是工具安装示例，不是依赖版本锁。具体复现应记录实际版本、路径和哈希。
GPU 环境还应记录 `nvidia-smi`、所选 `nvcc --version`、框架的实际 CUDA 版本，
不能把 `nvidia-smi` 显示的 CUDA 兼容版本当作已安装 Toolkit 的版本。

源码默认 `CMAKE_CUDA_ARCHITECTURES=120`，参考验证平台为 sm120。CMake 接受一个
数值架构，最低检查为70；其它架构会给出非参考平台警告。
**改为80只选择 helper 的编译目标，不会自动补齐 A100 的完整 HBF 接入。**
原版 gate 的运行库域为 CUDA12/13，没有另一分支的 CUDA11 适配。
详见[架构兼容说明](reference/cuda-architecture-compatibility.md)。

## 2. 获取代码和固定依赖

新目录克隆：

```bash
git clone --branch original https://github.com/Flash-sheep/HBFSim.git HBFSim-original
cd HBFSim-original
git submodule update --init --recursive -- third_party/bpftime third_party/mqsim
git submodule status -- third_party/bpftime third_party/mqsim
```

已有仓库先保存自己的修改，再 `git switch original`。只初始化上述两个构建子模块；
`paper` 是私有 Overleaf 子模块，不参与构建。

本版本在[依赖检查](../cmake/Dependencies.cmake)中固定：

- bpftime：`ec26daecc8e787fb80fd95dd596a576404a5e36e`。
- MQSim：`51f0f2d3fed92d88ef4a0fa61a38024b07bf9d16`。

不要直接更新到各自最新提交来绕过版本检查。构建目录和运行输出使用新目录，
模型、环境、二进制和原始结果不提交 Git。

## 3. CPU 构建和运行：不需要 CUDA 或模型

以下命令均从仓库根目录执行：

```bash
HBFSIM_REPO="$PWD"
HBFSIM_RUN_ROOT="$HBFSIM_REPO/../hbfsim-original-run"
HBFSIM_CPU_BUILD="$HBFSIM_RUN_ROOT/build-cpu"
mkdir -p "$HBFSIM_RUN_ROOT/tmp-cpu"

cmake -S "$HBFSIM_REPO" -B "$HBFSIM_CPU_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DHBFSIM_ENABLE_CUDA=OFF \
  -DHBFSIM_ENABLE_MQSIM=ON \
  -DHBFSIM_ENABLE_LLM_TESTS=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON

cmake --build "$HBFSIM_CPU_BUILD" --parallel 2
CUDA_VISIBLE_DEVICES='' TMPDIR="$HBFSIM_RUN_ROOT/tmp-cpu" \
  ctest --test-dir "$HBFSIM_CPU_BUILD" --output-on-failure
```

CPU 路径关闭 PkgConfig 是已有的环境规避方式：避免发现 libbpf 后进入不完整的
CUDA probe 构建路径，不会关闭 MQSim。GPU 构建不要照搬这个选项，否则会缺少
attach loader。见[原有环境说明](eval/runbooks/environment_setup.md)。

运行 CPU 媒体模拟示例：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$HBFSIM_CPU_BUILD/hbf_mqsim_bench" \
  --profile "$HBFSIM_REPO/configs/profiles/nominal.json" \
  --requests 1024 --bytes 16384 --operation read --arrival-gap-ns 0
```

输出是 JSON。检查 `requests.submitted` 与 `requests.completed`；
`timing_ns.modeled` 是模拟设备时间，`timing_ns.wall` 是软件执行耗时，不能混用。
`nominal.json` 是既有工程/合成 profile，不能据此声称已标定真实 HBF 器件。
CPU 媒体模拟成功也不证明 GPU 插桩已经工作。

完整测试的已知失败和历史结果见[TODO](../TODO.md)及主 README；应保存本机实际
CTest 输出，不能把文档中的历史通过数当作当前机器测试结果。

## 4. CUDA 构建：先确认工具链和目标 GPU

下面是 **sm120 参考目标的构建模板**。应修改 Toolkit 和兼容主机编译器路径，
并确认 GPU 确实对应目标架构；当前机器的 A100 不适用 sm120 运行命令。

```bash
HBFSIM_REPO="$PWD"
HBFSIM_RUN_ROOT="$HBFSIM_REPO/../hbfsim-original-run"
HBFSIM_GPU_BUILD="$HBFSIM_RUN_ROOT/build-cuda"
HBFSIM_TOOLKIT=/usr/local/cuda-12.8
HBFSIM_HOST_CXX=/usr/bin/g++-12

"$HBFSIM_TOOLKIT/bin/nvcc" --version
"$HBFSIM_HOST_CXX" --version

cmake -S "$HBFSIM_REPO" -B "$HBFSIM_GPU_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DHBFSIM_ENABLE_CUDA=ON \
  -DHBFSIM_ENABLE_MQSIM=ON \
  -DHBFSIM_ENABLE_LLM_TESTS=OFF \
  -DHBFSIM_ENABLE_TIMING_FUTURES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCUDAToolkit_ROOT="$HBFSIM_TOOLKIT" \
  -DCMAKE_CUDA_COMPILER="$HBFSIM_TOOLKIT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$HBFSIM_HOST_CXX" \
  -DCMAKE_CXX_COMPILER="$HBFSIM_HOST_CXX"

cmake --build "$HBFSIM_GPU_BUILD" --parallel 2
```

编译本身不运行 GPU workload；CUDA 构建的 CTest 可能包含真实 GPU 测试，不能把
全套 `ctest` 当作纯编译检查直接执行。[bootstrap.sh](../scripts/bootstrap.sh)也能配置，
但 CUDA 开启时硬编码检查 `/usr/local/cuda-12.8/bin/nvcc`；需要自选路径时使用上述
显式 CMake 命令。首次配置可能下载 FetchContent 依赖。

GPU 启动至少需要以下产物：

```text
build-cuda/libhbfsim_launch_gate.so
build-cuda/libptxpass_hbf.so
build-cuda/hbfsimd
build-cuda/hbfsim_bpftime_attach_loader
build-cuda/coverage_probe.bpf.o
build-cuda/benchmarks/cuda/hbf_microbench
```

其中 loader/probe 由 libbpf 与支持 BPF 的 Clang 检测启用。缺少它们时应补齐该
构建依赖并在独立目录重新配置，而不是跳过 readiness 检查。

## 5. 构建与原版 wrapper 配套的 bpftime

原版 wrapper 要求固定 ec26 子模块、`0001-exact-module-load-provenance.patch`
和 `bridge_version=2` 的构建 provenance。使用配套脚本，不要混用另一条构建流程
的补丁、共享库或自行填写 provenance。

```bash
bash scripts/build_patched_bpftime.sh --check

HBFSIM_AGENT_BUILD="$HBFSIM_RUN_ROOT/build-bpftime"
HBFSIM_CUDA_ROOT="$HBFSIM_TOOLKIT" HBFSIM_BUILD_JOBS=2 \
  bash scripts/build_patched_bpftime.sh "$HBFSIM_AGENT_BUILD" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$HBFSIM_TOOLKIT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$HBFSIM_HOST_CXX" \
  -DCMAKE_CXX_COMPILER="$HBFSIM_HOST_CXX"
```

脚本在构建目录内复制并打补丁，不修改原 bpftime 子模块。若本机 LLVM/Boost/CUDA
不在默认搜索路径，可通过该脚本尾部的 CMake 参数指定 `LLVM_DIR`、`BOOST_ROOT`
等实际位置。它不保证任意版本组合都可构建，应保留配置日志和实际产物哈希。

`scripts/reproduce_bpftime_agent.py` 与 `scripts/bpftime_build_lock/LOCK.json`
属于另一套精确锁定的历史复现流程，含原机器绝对路径和不同补丁身份；不能把其
输出直接冒充本 wrapper 所需构建。构建 provenance 不匹配时应停止核对，不删检查。

## 6. GPU 运行模板：baseline 与 HBF Timing 分开

本节是运行方法，不是本次已经执行的实验。新增 GPU 负载须先按
[实验确认契约](eq3_thermal/experiment_approval.md)冻结设备、workload、准确命令、
资源和停止条件，并取得该版本批准。本文编写和推送没有启动 GPU，也不更改驱动。

选择已经核实的空闲 GPU UUID，并为每次运行使用新输出目录：

```bash
export CUDA_VISIBLE_DEVICES=GPU-替换为已确认的UUID
HBFSIM_CASE=$(mktemp -d "$HBFSIM_RUN_ROOT/microbench.XXXXXX")
mkdir -p "$HBFSIM_CASE/baseline-report" "$HBFSIM_CASE/timing-report"
```

原生 baseline 不经 bpftime，也不注册 HBF 范围：

```bash
"$HBFSIM_GPU_BUILD/benchmarks/cuda/hbf_microbench" \
  --mode baseline --pattern sequential --bytes 65536 --iterations 16 --seed 1 \
  --profile "$HBFSIM_REPO/configs/profiles/nominal.json" \
  --report-dir "$HBFSIM_CASE/baseline-report" \
  --output "$HBFSIM_CASE/baseline.json"
```

HBF FAST Timing 使用配套 wrapper：

```bash
HBFSIM_BUILD_DIR="$HBFSIM_GPU_BUILD" \
HBFSIM_BPFTIME_BUILD_DIR="$HBFSIM_AGENT_BUILD" \
HBFSIM_CUDA_ROOT="$HBFSIM_TOOLKIT" \
HBFSIM_DAEMON_PATH="$HBFSIM_GPU_BUILD/hbfsimd" \
HBFSIM_INSTRUMENTATION_POLICY=strict \
BPFTIME_GLOBAL_SHM_NAME="hbfsim_original_$(basename "$HBFSIM_CASE")" \
BPFTIME_LOG_OUTPUT="$HBFSIM_CASE/bpftime-runtime.log" \
HBFSIM_COVERAGE_PATH="$HBFSIM_CASE/coverage.jsonl" \
HBFSIM_PASS_MANIFEST_PATH="$HBFSIM_CASE/pass-manifests.jsonl" \
  bash "$HBFSIM_REPO/scripts/run_with_bpftime.sh" -- \
  "$HBFSIM_GPU_BUILD/benchmarks/cuda/hbf_microbench" \
  --mode fast --pattern sequential --bytes 65536 --iterations 16 --seed 1 \
  --profile "$HBFSIM_REPO/configs/profiles/nominal.json" \
  --report-dir "$HBFSIM_CASE/timing-report" \
  --output "$HBFSIM_CASE/timing.json"
```

wrapper 会核对库和 provenance、在上述任务 namespace 中启动 loader、等待 attach-ready，
再运行目标并校验插桩激活记录。不要对已有结果路径重复运行：wrapper 会清理其
指定的 coverage/pass-manifest 文件。库依赖不在动态加载器搜索路径时，应仅在该
命令的环境中提供经核对的 `LD_LIBRARY_PATH`，不要全局预载 agent。

验收需同时核对：baseline/Timing 数值结果、实际 modeled coverage、模块/PTX/原生
函数身份绑定、普通服务请求完成及错误计数。进程退出0、profile已读取或地址范围
已注册，均不能单独证明全部访问已由 HBF 建模。

## 7. Llama2 第15层及常见阻塞

该原版分支没有 `sm_80支持` 分支中的私有 CUDA11 环境、适配包装器、模型或
已构建二进制。Llama2 的层号15、batch/query/context=1/1/32、九参数范围和输入
身份说明见[原始单层交接](llama2_layer15_live/BLOCKERS.md)。完整层接入还需冻结
原 workload、模型 tensor、实际库、PTX/container/entry 和新进程地址；原 trace
或旧进程的 arena 地址不能直接用于新机器。

| 现象 | 应检查的内容 |
| --- | --- |
| 找不到 CMake/OpenSSL/固定子模块 | 实际 PATH、开发头文件、上述子模块 gitlink，先完成 CPU 配置 |
| CUDA版本拒绝 | 原版构建下限12.8，不用改版本判断绕过 |
| A100原生算子能跑但无法HBF插桩 | matching cubin 可支持原生运行；HBF还需要原始兼容PTX和接入运行库域 |
| cuBLAS PTX只有sm90或Torch没有PTX | 调查实际库容器；不能靠CUDA版本号推断PTX，更不能直接把target改成sm80 |
| provenance mismatch / attach timeout | 检查配套补丁、产物路径、libbpf/Clang、loader/probe及原始日志 |
| unsupported指令或coverage缺失 | 保留失败并定位；不当作无HBF访问而静默放行 |

以上命令按本分支源码/CLI静态核对。本次文档变更没有重新构建或运行 CPU/GPU
实验；本机是否通过必须以另存的实际执行收据为准。
