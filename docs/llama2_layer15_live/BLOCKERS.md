# Llama2 单层 HBF 运行阻塞与迁移说明

截至 2026-10-19，Llama-2-7B 第 15 层、context32 的 CPU 输入校验已通过，但真实 GPU 加 HBF Timing 运行尚未启动。当前机器的 CUDA 工具链和已安装算子库的 PTX 不满足本次接入条件。换机器后需要重新核对工具链、实际 kernel 的 PTX 身份和动态覆盖，不能直接沿用旧机器的就绪性或地址。

本次检查基于 HBFSim `d2e6ce239c117c640fe68c8b5b2a7c77cf184da6`。本目录保存说明与原始检查收据，不包含模型权重、运行二进制，也不是已批准的实验包。

## 工作负载与复用目标

目标是直接执行原 trace 背后的单层 workload，而非重放 Accel-Sim SASS trace。

| 参数 | 本次冻结值 | 证据身份 |
| --- | --- | --- |
| 模型 | 本地 Llama-2-7B checkpoint | 原配置与 checkpoint 元信息 |
| 层 | layer index 15 | 原采集脚本与权重 manifest |
| batch / query / context | 1 / 1 / 32 | 原采集脚本与权重 manifest |
| 历史 KV 长度 | 31 | context 减去 query |
| 精度 | FP16 | 原脚本与 checkpoint |
| hidden / intermediate | 4096 / 11008 | 模型 config |
| attention / KV heads | 32 / 32 | 模型 config |
| seed / warmup | 1 / 0 | 沿用原脚本默认值的 pilot 提案；不是旧 capture 独立验证过的种子元信息 |
| 注册范围 | 9 个参数 tensor，404,766,720 B | 原 weight arena manifest |
| checkpoint 中本层条目 | 10 个，404,766,848 B | 本次 CPU 校验与逐 tensor 哈希 |

两个计数的差异是额外的 128 B `self_attn.rotary_emb.inv_freq` 旧 buffer，不能把它当作额外注册参数。真实层权重来自 checkpoint；hidden states 和历史 K/V 沿用原脚本的固定种子随机构造。这是单层 forward，不是完整模型生成或真实 token 序列的逐层输入。

原 trace 有 55 个 kernel；在另一套硬件和库上重新运行时，kernel 数量、实现与地址可能变化，不能预设仍然是相同的 55 个 kernel。

## 已完成的检查

- **CODE_OBSERVED**：两份原 workload 脚本和小型 JSON 输入已在旧机任务目录冻结；文件身份及所需本层 checkpoint tensor 的原始字节哈希见 [input_identity.json](evidence/input_identity.json)。本 Git 归档保留 JSON 快照与脚本哈希，未复制 donor Python 源码。
- **实际 CPU 收据**：使用 `cloudos` Python 执行原脚本的 `--validate-only`，退出码 0，输出 `weights=10 fp16_bytes=404766848`。进程环境为 `CUDA_VISIBLE_DEVICES=''`，没有运行 GPU forward。见 [CPU 输入校验](evidence/cpu_input_validation.json)。
- **实际配置收据**：保留初始 CMake 失败和补齐本地 CMake 后的失败，未修改生产源码或放宽项目版本要求。
- **实际静态清点**：对当前 cuBLAS 与 Torch CUDA 容器执行 `cuobjdump --list-ptx`，未执行 GPU kernel。
- **执行状态**：GPU workload 启动数为 0；未运行 native baseline、FAST 或 REFERENCE，也未创建用户批准记录。见 [状态快照](evidence/STATUS.json)。

## 当前阻塞

| 项目 | 旧机器观测 | 结论与证据 |
| --- | --- | --- |
| 系统 CMake | 3.22.1；项目要求至少 3.25 | 已通过任务目录内 CMake 3.31.10 消除该项；[首次失败](evidence/configure_probe.json)保留 |
| CUDA Toolkit | 12.1；项目明确要求至少 12.8 | 阻塞 CUDA 构建；[再次配置失败](evidence/configure_probe_cmake331.json)明确记录错误 |
| GPU | Tesla V100-SXM2-16GB，`sm_70`，driver 565.57.01 | CMake 将 `sm_70` 标为非参考平台，不能把成功构建当作协议已验证 |
| `libcublas.so.12` | 191 个嵌入 PTX 成员，目标全部 `sm_90` | 现有 PTX 不能作为本机 `sm_70` 的插桩入口 |
| `libcublasLt.so.12` | 95 个嵌入 PTX 成员，目标全部 `sm_90` | 同上 |
| `libtorch_cuda.so` | 当前 `--list-ptx` 返回未找到嵌入 PTX | 该容器没有提供本次所需的嵌入 PTX 来源 |
| core / gate / daemon / agent | 未构建、无本次已绑定可执行产物 | 尚无可提交的 HBF 运行命令和二进制身份 |
| GPU 实验预检 | `PENDING_USER_APPROVAL`，执行就绪性 `BLOCKED` | 当前草案不具备提交条件 |

PTX 完整列表和摘要分别见 [ptx_inventory.json](evidence/ptx_inventory.json) 与 [host_audit.json](evidence/host_audit.json)。

这些发现不表示原生 CUDA 算子不能在 V100 上运行：算子仍可能通过匹配 cubin 执行。阻塞的是 HBFSim 所需的可审查、兼容目标架构且可插桩的 PTX 路径。也没有据此断言所有其它库或其它环境都没有适用 PTX。

不能简单修改 PTX 的 `.target` 来声称兼容。换成 `sm_90` 或项目参考架构 `sm_120` 的机器，也仍需核对实际选用 kernel 的 entry、模块身份、原生与 staged 绑定和动态覆盖；仅仅硬件架构数值匹配不代表问题自动解决。

## 新机器上的继续步骤

1. **准备源码与外部输入。** 获取本仓库和固定子模块；另行搬运 Llama checkpoint 及原始两份 workload 脚本。脚本原位置及 SHA256 保存在 `input_identity.json`。新路径可以变化，内容身份要重新核对。不要搬运旧 arena 基址用于新进程注册。
2. **记录实际环境。** 保存 GPU 型号、UUID、driver、CUDA Toolkit、CMake、编译器、Python、Torch、Transformers 和 safetensors 版本及资源余量。旧机 Torch 2.4.0、Transformers 4.57.1、safetensors 0.6.2 是来源信息，不是所有新机器已验证的兼容组合。
3. **先做 CPU 准备。** 使用 CMake 至少 3.25、CUDA Toolkit 至少 12.8，按目标 GPU 设置单一数值 `CMAKE_CUDA_ARCHITECTURES`。独立构建 core、gate、daemon、pass 和匹配的 patched bpftime agent，不修改 driver、系统环境或公共接口。Toolkit 下限通过不代表 driver 和所有依赖已兼容，应继续记录实际构建与装载检查。
4. **检查实际 kernel PTX。** 从新环境的真实 Torch/cuBLAS 等容器提取和校验原始 PTX，冻结 entry 与容器哈希，完成变换及 CPU 编译检查。不得直接继承 Qwen3/Blackwell 历史绑定或旧 V100 清点结果。
5. **完善原单层接入。** 复用原脚本的 checkpoint 加载、输入生成和 weight arena 布局，在新进程中注册实际 arena，接入匹配的 launch gate。该适配器和完整运行包装当前尚未交付，不能把原采集脚本直接当作已接入 HBFSim 的 launcher。
6. **重新冻结执行版预检。** 写入新设备、输入、构建产物、PTX 绑定、准确命令、资源与停止条件，经项目规则要求的具体版本确认后启动。迁移文档、历史授权和本次 Git push 均不替代 GPU 实验批准。
7. **先运行有界 pilot。** 原生 baseline 与 HBF Timing 对照分别保存输出数值、CUDA event 与墙钟、逐参数 addressed/active modeled 覆盖、未覆盖字节、请求完成及错误。任何未覆盖权重都要明确报告，不能凭 workload 成功声称完整层 HBF 行为。

子模块同步入口与本次 CPU 配置模板如下。配置模板不会启动 GPU；根据新机器替换路径与架构。`paper` 为非构建所需私有子模块，不初始化。

```bash
git submodule sync --recursive -- third_party/bpftime third_party/mqsim
git submodule update --init --recursive -- third_party/bpftime third_party/mqsim

cmake -S /path/to/HBFSim -B /path/to/new-build -G Ninja \
  -DHBFSIM_ENABLE_CUDA=ON \
  -DHBFSIM_ENABLE_MQSIM=ON \
  -DHBFSIM_ENABLE_TIMING_FUTURES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=YOUR_NUMERIC_GPU_ARCH \
  -DCUDAToolkit_ROOT=/path/to/cuda
```

CUDA 编译不需要执行 GPU workload。新 GPU 若要启用 timing futures，属于另一项兼容性与范围要求；本 pilot 提案保持关闭。

## 预检提案与结论边界

[原预检文本](evidence/PREFLIGHT.txt) 和 [机器可读草案](evidence/preflight_manifest.json) 逐字节保留。

草案计划 native 一次、instrumented FAST 一次，REFERENCE 和参数扫描延后；没有已验证的本路径 zero-wait 臂。Timing 使用项目 nominal 媒体情景的 16 KiB 页、10 μs read、100 μs program、512 GB/s、QD128，复制 profile 中仅修改名称与 `time_scale=1`。该选择尚未获执行批准，不等价于原 FLINT 的 4 KiB 页、2 MiB burst 与 PB/CB pipeline。

几何、功率、热网络、可靠性和维护不是本 pilot 的验证目标；热/可靠性为 NOT_USED。未知产品参数不补成零。实际 GPU 时间需要与模拟器开销分开解释。

草案中的 16 GiB 主机 RAM、6 GiB 显存、12 GiB 磁盘、15 GPU 分钟及 watchdog 是按旧机观察提出的有限资源方案，不是项目永久边界，也不是新机器已批准预算。新机器应重新评估。旧报告中的 `gpu_workload_launched=false` 只描述本次检查。

用户仅要求归档并推送阻塞材料；**未接受替换算子实现**。若原算子找不到适用 PTX，需要另行确认替换算子、SASS lifting 或其它执行方案，不能自行替换后当作原 kernel 路径复现。

## 归档与校验

- [ARCHIVE_INDEX.json](evidence/ARCHIVE_INDEX.json) 列出归档文件大小与 SHA256。
- `evidence/` 保存原始内容，绝对路径、旧机 UUID 和原始错误信息未改写；它们是历史来源，不是新机默认配置。
- 规划清单内容哈希为 `6f3d6d350f4ca73e6c73efc3d3c59771de571d676c53b5a543dd83f1e7c8236c`。其算法见清单，**不是** EQ3 gate 的可执行 `canonical_manifest_hash`；阻塞项不允许被伪报为 PASSED。
- 未归档模型 payload、原 trace、工具链、编译产物和 donor Python 源码。源脚本/准备脚本的哈希留在原清单中，不能因此声称本 Git 目录已包含完整复现包。
- 原始任务目录仍保留在 `/home/data/long/hbfsim-runs/llama2-layer15-ctx32-live-v1-20261019`，其中含源码快照、本地 CMake 和配置探针工件。

GPU 实验批准规则见 [`AGENTS.md`](../../AGENTS.md) 和 [`experiment_approval.md`](../eq3_thermal/experiment_approval.md)。
