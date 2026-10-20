# A100 sm80 PTX 样本实际 GPU 结果

2026-10-19，获用户具体版本确认后，已在 GPU1 A100 上完成三组、各一次启动。
结果为 **PASS**，三组输出逐字节相同，2048个 FP32 全部清零，两侧256B guard 完整。

| 组 | 输入路径 | kernel启动数 | 输出/guard |
| --- | --- | --- | --- |
| raw77 | CUDA11.7原始PTX7.7 → sm80 cubin | 1 | PASS |
| raw87 | 仅将版本行调整至8.7 → sm80 cubin | 1 | PASS |
| staged87 | 同一entry插入当前helper，版本8.7 → sm80 cubin | 1 | PASS |

三组8704B输出（包含guard）的共同SHA256：
`7b85d3fa26b3b390f124c4e659969274a561705b8d2c036ada43e7b60fae88df`。
独立读取输出文件后的[字节复核](current_machine_20261019/gpu_probe_v1/post_run_byte_check.json)也通过。
完整进程墙钟为1.118190831秒；各组记录的launch+sync墙钟不是CUDA event测量，
不能据此比较性能。

## 执行身份与批准

- experiment_id：`LLAMA2-SM80-PTX-GPU-PROBE`；version：`1`。
- canonical_manifest_hash：`a54ce76c6ee340533c953a407d126f44b257014b61481e27be8ffc797e5c297f`。
- [实际用户确认](current_machine_20261019/gpu_probe_v1/USER_CONFIRMATION.json)与
  [原文](current_machine_20261019/gpu_probe_v1/USER_SOURCE.txt)保留；
  [门禁检查](current_machine_20261019/gpu_probe_v1/approved_gate.json)通过。
- GPU UUID：`GPU-5302a497-ea8c-bb7b-7235-f7464792093e`，A100-PCIE-40GB，
  sm80，driver565.57.01。worker PID3194461，parent PID3194446。
- 使用CUDA12.8 ptxas预先编译的cubin，经CUDA Driver API直接装载、查询函数及执行。
  LOCAL entry查询实际成功；不是运行时PTX JIT验证。
- 前台工具会话45837已退出0，watchdog无停止原因，`execution-v1/DONE`存在。
  GPU1结束后回到135MiB已用显存、利用率0、43°C的观测状态。
  未执行GPU0任务、driver修改、CUDA库替换、push或merge。

## 结论边界

实际测试的是cuBLAS scal清零分支的工程输入，单block、256线程、32x64 FP32。
不是Llama第15层的实际选用kernel，也不是FP16完整forward。

staged87的`__hbfsim_control`与generation均在启动前检查为0，走
**UNBOUND_PASSTHROUGH**。该结果证明当前样本可以在不改原CUDA库文件的情况下，
通过独立PTX副本升版、插入现有helper、编译并在A100执行，且本输入的输出相同。
未验证HBF地址注册、modeled请求/延迟、MQSim、实际bpftime/gate绑定、其它输入或kernel。
既有5条ld.param unsupported记录、旧失败和原归档均保留。

完整输入、二进制、原始输出及看护记录在
`/home/yjl/HBFSim/artifacts/llama2-sm80-gpu-probe-v1`；
其中`ARCHIVE_INDEX.json`列出文件SHA256。
Git文档目录保存[实际结果](current_machine_20261019/gpu_probe_v1/RESULT.json)、
[结束状态](current_machine_20261019/gpu_probe_v1/FINISH.json)和
[最新状态](current_machine_20261019/gpu_probe_v1/STATUS.json)等小型副本，未归档模型或cubin。
冻结的manifest保留批准前的状态字段作为原始预检快照；批准与完成状态以独立记录为准。
