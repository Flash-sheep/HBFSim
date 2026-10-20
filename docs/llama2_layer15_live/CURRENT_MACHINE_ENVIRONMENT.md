# 当前机器实验环境配置（2026-10-19）

最新完整层准备见[FULL_LAYER_PREPARATION.md](FULL_LAYER_PREPARATION.md)：用户提供的原始输入已找齐，CPU完整层前向通过；GPU/HBF Timing仍待接入与新预检。

最新状态：用户批准后，三组sm80样本已在GPU1各运行一次并全部通过。
见[实际GPU结果](GPU_SM80_SAMPLE_RESULT.md)。验证范围为未绑定HBF控制区的
工程样本，不代表Llama或HBF Timing通过；下面的零GPU/PENDING描述属于各次准备时点。

本记录对应用户“在当前机器上寻找可用的 conda 环境并进行实验环境配置”的要求。
这是隔离环境配置与固定 CPU 软件验证，不是 GPU 实验预检批准；原
[BLOCKERS.md](BLOCKERS.md) 及其 `evidence/` 历史收据保持不变。

## 环境选择与位置

实际清点发现以下已有 conda 环境，完整收据见
[EXISTING_CONDA_ENVIRONMENTS.json](current_machine_20261019/EXISTING_CONDA_ENVIRONMENTS.json)。

| 环境 | 实际结果 |
| --- | --- |
| `/home/fyf/anaconda3` | Python 3.9.13、Torch 2.5.1+cu124、Transformers 4.47.1、safetensors 0.4.5；CPU 导入可用，但同时触发已有 SciPy/NumPy、TensorFlow 相关警告 |
| `envs/DNN`、`envs/GNN` | conda 有记录，但不存在可执行的 `bin/python` |
| `envs/gnn` | Python 3.8.20，无所需 Torch/Transformers/safetensors |
| `envs/gnn-1` | Python 3.9.21，无所需 Torch/Transformers/safetensors |
| `/home/yjl/.conda/envs/litellm311` | Python 3.11.15，无所需 Torch/Transformers/safetensors |

USER_CONFIRMED：配置本机实验环境。INFERRED：创建独立环境，避免修改共享
base 和其它应用环境。DOC_DERIVED：沿用归档中的三个主要 Python 包版本，
并满足项目 CUDA Toolkit 下限；版本一致不代表 GPU 路径已经复现。

任务目录：`/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1`。
初始 `/home` 可用约 13 GiB，该目录所在分区可用约 186 GiB，故下载缓存、
环境和构建均放在此处；checkpoint 原地只读复用。该位置是本机临时目录，
可能受系统清理策略影响，不是永久归档位置。

| 组成 | 本次配置 |
| --- | --- |
| Python conda prefix | 任务目录下 `env/`；Python 3.11.17 |
| Python 主要依赖 | Torch 2.4.0、Transformers 4.57.1、safetensors 0.6.2、NumPy 1.26.4 |
| Torch 自带 CUDA 运行库 | 12.1；与 HBFSim 编译 Toolkit 分别记录 |
| CUDA conda prefix | 任务目录下 `cuda/`；nvcc 12.8.93、cudart 12.8.90，含 ptxas、cuobjdump、NVRTC、nvPTXCompiler 静态库 |
| CMake | 任务私有 3.31.10；系统版本未修改 |
| C++ 编译器 | `/usr/bin/g++-11`，11.5.0 |
| 构建选项 | 单一数值架构 `80`，MQSim ON，timing futures OFF，LLM tests OFF |
| gate 链接配置 | 仓库现有 `cmake/cudart-versions.map` |
| 并发 | 构建 2 jobs；OMP/OpenBLAS/MKL/NumExpr 各 1 线程 |

激活命令（只影响当前 shell）：

```bash
source /var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/activate.sh
```

激活脚本默认设置 `CUDA_VISIBLE_DEVICES=''`，并启用 Hugging Face 离线模式。
它不改 driver、系统 CUDA 链接或 shell 启动文件，也不设置全局动态库路径。
安装未修改已有 conda 环境。

## 已验证范围

- 本机设备为两张 A100-PCIE-40GB，`sm_80`，驱动 565.57.01；仅作只读枚举。
  初始 GPU0 已有其它进程占用，未申请或保留任何 GPU。
- checkpoint 位于 `/home/yjl/yjl-r760/llama2-7b-hf-chat`。配置、索引及第15层
  全部10个 tensor 与原归档 SHA256 一致；CPU 加载共 404,766,848 B。
  其中注册参数仍为9个、404,766,720 B，额外128 B 是旧 rotary buffer。
- CPU 验证包括 Torch、Transformers、safetensors、`LlamaDecoderLayer` 导入和
  checkpoint tensor 读取/哈希；没有执行 forward，也没有重建原脚本或伪称
  运行其 `--validate-only`。
- `pip check` 通过：`No broken requirements found.` 全部传递依赖另存固定版本
  清单与原始 pip 安装来源/下载哈希报告，不将三个主包版本相同称为完整环境相同。
- 已构建 `libhbfsim`、`hbfsimd`、`libhbfsim_launch_gate`、`ptxpass_hbf` 和
  `libptxpass_hbf`；所选6项 CPU 测试全部通过，动态依赖检查未发现缺失项。
  这不是完整项目测试套件验收。
- helper PTX 为 PTX 8.7、`sm_80`，通过 CUDA12.8 ptxas 的 CPU 离线汇编；
  不代表真实 Torch/cuBLAS kernel 的插桩覆盖或驱动装载通过。

首次 CMake3.25 配置因缺少 `CUDA::nvToolsExt` target 失败；随后使用私有
CMake3.31.10。首次 gate 链接因缺少 cudart symbol version node 失败；随后
通过现有配置项指定仓库版本映射文件。两份失败日志均保留，未修改生产源码。

## Llama / HBF Timing GPU 运行仍未就绪

GPU workload 启动数为0，native baseline、FAST、REFERENCE 均未执行。
GPU 实验状态仍为 `PENDING_USER_APPROVAL`，执行就绪性为 `BLOCKED`。

安装环境的静态 PTX 清点见
[PTX_INVENTORY.json](current_machine_20261019/PTX_INVENTORY.json)。它只覆盖列出的
三个库容器，不能替代实际选用 kernel 的 entry、模块身份和动态覆盖验证。
已有共享 base 的 cuBLAS 静态清点也仅发现 `sm_90` PTX。

| 新隔离环境容器 | 静态清点 |
| --- | --- |
| `libcublas.so.12` | 191个 PTX 成员，列表架构全部 `sm_90` |
| `libcublasLt.so.12` | 95个 PTX 成员，列表架构全部 `sm_90` |
| `libtorch_cuda.so` | 未发现嵌入 PTX |

这些已检查容器没有提供本机 `sm_80` 所需的可插桩 PTX 来源。匹配 cubin 的
原生执行能力与此不同；未据此断言原生 CUDA 无法运行，也未检查所有其它库。

### 后续只读补查：本机 CUDA11.7 有 sm_80 PTX 候选

用户询问能否下载其它版本后，补查本机 CUDA11.7/12.3/12.4 库。
CUDA11.7 的 `libcublas.so.11.10.3.66` 与 `libcublasLt.so.11.10.3.66`
分别列出231/88个 `sm_80` PTX 成员；各抽取一个原始成员，确认声明为
`.version 7.7`、`.target sm_80`。这提供了现成候选，不能将当前 cu12 环境的
缺口扩张为所有发行版本均无适用 PTX。CUDA12.3/12.4 所检查库仍仅列出 sm_90。

当前 Torch2.4.0 的 ELF 依赖明确为 `libcudart.so.12`、`libcublas.so.12`、
`libcublasLt.so.12`，所以这不是直接替换旧 `.so.11` 就能使用的组合。
需要核对匹配的 Torch/CUDA11 环境、实际被选中 kernel 的 PTX 身份，以及
旧 PTX7.7 与当前插件声明的 PTX>=8.7/helper 的编译兼容性。
抽样模块不代表 workload 实际选用的 kernel，未进行环境替换或 GPU 运行。
本配置的已验证状态不变，原冻结收据不改写；新增证据分别见
[候选库清点](current_machine_20261019/ALTERNATE_LIBRARY_INVENTORY.json)及
[原始PTX与Torch依赖](current_machine_20261019/ALTERNATE_PTX_AND_TORCH_DEPENDENCIES.json)。

### 固定 CPU 软件检查：旧 PTX 可提取，合并 helper 后有版本语法冲突

仅对上述已提取 cuBLAS 模块的首个 scal entry 做一次同步转换器 CLI 检查，
不是 Llama 实际选用 kernel，也不是 GPU 实验。原始 `.version 7.7`、
`.target sm_80` PTX 经 CUDA12.8 ptxas `--compile-only --gpu-name=sm_80`
通过；现有转换器成功改写12条访存，保留原版本和目标声明。
合并 helper 后的 PTX 编译失败：`Feature 'used_bytes_mask' requires PTX ISA
.version 8.3 or later`。该语法来自当前 helper，而模块仍声明7.7。
这是实际复现的 PTX 语言版本兼容问题，不能归因为 A100 不支持 sm_80。
未修改原始 PTX、版本声明、生产源码或插件最低版本配置。

转换器另报告5条 `ld.param` unsupported，完整计数保留；CLI 改写成功不代表
实际插件参数绑定、完整覆盖或模型数值通过。HBF 插件 `process_input()` 当前
不调用通用 PTX 最低版本校验，因此配置中的 `ptx_version_min=8.7` 不能被
当作本次拒绝点；实际失败发生在合并输出的 ptxas 编译阶段。
见[检查元信息](current_machine_20261019/LEGACY_SM80_CPU_PRECHECK.json)、
[完整结果](current_machine_20261019/LEGACY_SM80_CPU_RESULT.json)及
[原始错误](current_machine_20261019/LEGACY_SM80_CPU_ASSEMBLY_ERROR.txt)。
状态为 FAILED，未执行 GPU、模块装载、数值比较或性能测量。

### 独立 PTX 副本版本适配：CPU 编译通过，GPU 未运行

用户询问能否绕过声明且不影响 CUDA 库后，仅对上次同一原始/已插桩样本的
独立文件副本，将唯一 `.version 7.7` 行调整为 `.version 8.7`。
逐字节断言除此版本行外全部内容相同，`.target sm_80` 保持不变；未修改
转换器、helper、插件配置或安装库。两个副本经 CUDA12.8 ptxas
`--gpu-name=sm_80` 均成功生成 cubin，未删除 pragma 或屏蔽编译器错误。
原始 PTX、此前失败的 staged PTX、两份 CUDA11.7 cuBLAS 库的前后 SHA256 一致。

状态为 `CPU_COMPILATION_PASS_GPU_NOT_RUN`，只证明该样本经版本行适配后能编译。
语言主版本变化后的语义等价性、实际 kernel 身份、插件参数绑定、模块装载、
CUDA11/12 运行库及 gate 兼容性、数值和性能均未验证，不能视为 GPU 运行许可
或原 kernel 逐字节复现。旧 FAILED 收据不覆盖、不改写。
见[检查元信息](current_machine_20261019/LEGACY_SM80_VERSION_COPY_PRECHECK.json)、
[完整结果](current_machine_20261019/LEGACY_SM80_VERSION_COPY_RESULT.json)及
[固定检查脚本](current_machine_20261019/legacy_sm80_version_copy_check.py)。

### GPU 样本批准前准备（历史）

用户要求“尝试运行”后，准备了独立 Driver API 样本脚本、三组 cubin、固定参数、
CPU 地址/ABI 检查、未批准时 parent/worker 均先拒绝的门禁检查，以及 GPU UUID、
占用、内存、温度和60秒 watchdog。未启动 GPU，未创建用户批准记录。

预检：[PREFLIGHT.md](../../artifacts/llama2-sm80-gpu-probe-v1/PREFLIGHT.md)。
机器清单：[manifest.json](../../artifacts/llama2-sm80-gpu-probe-v1/manifest.json)。
experiment_id=LLAMA2-SM80-PTX-GPU-PROBE，version=1，状态 PENDING_USER_APPROVAL。
canonical_manifest_hash=`a54ce76c6ee340533c953a407d126f44b257014b61481e27be8ffc797e5c297f`。

拟在 GPU1/UUID `GPU-5302a497-ea8c-bb7b-7235-f7464792093e` 上比较原7.7、
仅升版8.7、升版并插桩三组，各启动一个kernel一次；预期32x64 FP32清零，
两侧guard完整。HBF control必须为0，仅验证未绑定时的passthrough，不验证
Llama、HBF Timing、MQSim或实际插件绑定；不装载Torch/cuBLAS运行库。
预计1–10秒、watchdog60秒，任务RAM2GiB/worker RSS1GiB、GPU已用显存1GiB、
磁盘128MiB/输出64MiB。用户确认后仍须通过当前文件/运行时身份和资源检查。
门禁当前按预期返回 PENDING_USER_APPROVAL/launch_performed=false。

### GPU 样本已获批准并实际完成

上述version1/hash已获真实用户明确确认，门禁通过后执行一次完整三组流程。
raw77、raw87、staged87各启动一次，均通过零输出与guard检查，输出SHA256一致。
进程墙钟1.118190831秒，worker退出0，DONE路径已写入，GPU1已释放。
原manifest和准备期PENDING记录不改写，实际确认、执行结果和scope边界另见
[GPU_SM80_SAMPLE_RESULT.md](GPU_SM80_SAMPLE_RESULT.md)。当前样本仅验证
UNBOUND_PASSTHROUGH；Llama/HBF Timing完整链路仍未验证。

更新：原始两份workload Python源码已从`/mnt/hbfsim-workload`找齐，内容哈希与归档一致；原脚本CPU validate-only及固定完整前向已通过。
匹配的 patched bpftime agent 尚未构建；历史构建锁绑定其它主机的路径、
工具版本及哈希，不能只替换路径便声称通过。单层 HBF 适配器与完整运行包装
也不属于此次环境配置的交付。

驱动565与实际 PTX/JIT/cubin 装载路径尚未验证。NVIDIA 说明 CUDA 同主版本
兼容存在 PTX 限制；本项目也有 nvPTXCompiler 先编译 cubin 的路径，因此
不能单凭驱动版本或离线构建结果判定实际运行成功或失败。
参见 [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

后续须冻结实际 kernel 绑定、agent 与 launcher，再重新评估本机资源，形成
准确命令和停止条件齐全的执行版预检，取得用户明确确认后才能启动 GPU 实验。
未替换原算子，也未修改公共 ABI、PTX/TMA/future/cache 或原始结果。

## 复用记录

[机器可读配置清单](current_machine_20261019/ENVIRONMENT_MANIFEST.json) 汇总版本、
身份、产物哈希、验证和剩余阻塞。新目录仅保存本次配置记录，不包含模型或二进制。
完整安装日志、pip 下载来源/哈希、构建日志和失败日志留在本机任务目录。

- [Python conda 显式锁](current_machine_20261019/conda-python-explicit.txt)
- [CUDA conda 显式锁](current_machine_20261019/conda-cuda-explicit.txt)
- [pip 固定版本清单](current_machine_20261019/requirements.lock.txt)
- [CPU 输入验证收据](current_machine_20261019/CPU_VALIDATION.json)
- [构建收据](current_machine_20261019/BUILD_RECEIPT.json)
- [离线汇编收据](current_machine_20261019/ASSEMBLY_VALIDATION.json)
- [激活脚本副本](current_machine_20261019/activate.sh)
- [固定构建及 CPU 检查脚本副本](current_machine_20261019/build_cpu.sh)

这些记录绑定本机路径。迁移时应重新核对输入、工具链、设备、资源及实际消费者；
不要复用旧进程的 arena 基址，也不要将本配置记录视为 GPU 批准文件。
