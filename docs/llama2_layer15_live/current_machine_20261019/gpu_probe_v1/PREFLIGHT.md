# A100 sm80 PTX 样本 GPU 预检 v1

状态：PENDING_USER_APPROVAL。GPU 尚未启动，未创建用户确认记录。

- experiment_id=LLAMA2-SM80-PTX-GPU-PROBE
- version=1
- canonical_manifest_hash=a54ce76c6ee340533c953a407d126f44b257014b61481e27be8ffc797e5c297f
- 机器可读清单：[manifest.json](manifest.json)
- 已准备脚本：[probe.py](probe.py)，默认模式仅 CPU 检查。

## 问题、证据与结论范围

USER_CONFIRMED：用户要求“尝试运行”；尚未确认本具体版本。
DOC_DERIVED：现有 CUDA11.7 cuBLAS 样本、CPU 插桩/编译与版本副本结果。
INFERRED：先对刚才的样本做最小 GPU 工程验证，不启动尚未就绪的 Llama 全层。

问题：原 PTX7.7、仅升至8.7、升至8.7并插桩后，能否在 A100 上装载并执行，
产生同样的预期输出，且不损坏 guard？证据仅为 GPU_ENGINEERING_FIXTURE。
不是 Llama2 第15层、实际 Llama 选用的 GEMM、HBF Timing、MQSim/吞吐、物理热、
可靠性、维护或性能实验；不产生 MODEL_FREEZE，不清除历史失败。

该样本是 cuBLAS scal 的清零分支，FP32。三组使用同一个 entry，未替换算子实现。
原模块包含其它 entry，但只启动已审查的这一项。ELF 中 entry 为 LOCAL，驱动
能否查到仍待本次验证；若查找失败立即停止，不擅自改变可见性或重试。

## 代码、环境与设备

代码 HEAD：874a20f289b7bd587fa4048a788e12df2618444b。
使用已绑定的三份 cubin 和固定参数；Python3.11.17 ctypes 直接使用 CUDA Driver API。
编译工具为 CUDA12.8.93、目标 sm80；运行时不装载 Torch、cuBLAS、libcudart、
bpftime 或 launch gate，避免将这个小测试误称为完整接入验证。
真实 Python、libcuda 文件哈希在启动前后核对，脚本、输入、helper 和转换器源码
通过项目 gate 校验。原 .so、原始 PTX、失败收据保持不变。

固定设备：GPU1，NVIDIA A100-PCIE-40GB，sm80，driver565.57.01。
UUID：GPU-5302a497-ea8c-bb7b-7235-f7464792093e。
GPU0 不使用，当前已有其它用户任务。UUID、驱动、架构与占用将在启动时重查。
使用 advisory lock；尚未持有 GPU 预约，不声称其它软件受该锁约束。
几何/材料/热边界/可靠性参数 NOT_USED；器件功率限制只记录现状，不修改。

## workload、初态和对照

- 一个 block，256线程，grid=(1,1,1)，block=(256,1,1)，shared=0。
- 参数0为48B结构，其 uint32 字段为 [0,0,64,32,0,64,0,0,0,0,0,0]。
  来自已检查原 PTX 的8/12/20/40字节字段和访存地址推导，不是产品参数。
- 输出32x64，共2048个FP32，初始全1.0；预期全部清零。
- 分配8704B，两侧各256B的0xa5 guard。三指针参数均指向中央输出区；
  此 entry 实际仅消费指针参数2。
- 无随机输入，seed NOT_USED；无 warmup，每组新装载模块、重置数据。
- raw77：原PTX7.7编译的sm80 cubin。
- raw87：仅版本行升至8.7，其它字节不变。
- staged87：同一entry加入现有同步helper，版本8.7，目标仍sm80。
- 三组各1次，共最多3次kernel launch；串行、无扫描、无跨设备、失败不自动重试。
- staged 的 __hbfsim_control / generation 必须均为0：明确是 UNBOUND_PASSTHROUGH，
  不注册 HBF 地址、不启动 daemon、不产生 modeled requests 或延迟。
- 既有 CLI 的5条 ld.param unsupported 记录保留，本测试不豁免覆盖缺口。

## 资源、耗时与停止条件

模拟时长 NOT_USED。实际 wall 预计1–10秒，尚无本机GPU测量；总 watchdog60秒，
GPU预算1分钟，终止宽限5秒。资源是本测试新评估的有限计划，不恢复历史固定上限。

任务主机RAM预算2GiB，worker RSS超过1GiB停止；GPU总已用显存达到1GiB停止，
包含上下文开销，实际显式buffer仅8704B；任务磁盘预算128MiB、输出上限64MiB。
CPU配置1、执行1、CPU预算0.02小时，OMP/BLAS/MKL1，运行时不编译。
保留主机可用RAM>=4GiB、文件系统空闲>=1GiB。

启动要求：目标GPU无其它计算进程、利用率0、已用显存<1GiB、温度<75°C。
执行中温度>=80°C、外来进程进入、资源越界、任何CUDA/API/符号错误、输出或guard
错误均停止后续组。75/80°C是本测试的 SCENARIO_ASSUMPTION 安全条件，不是厂商规格。
进程内部每1秒自检；不需要AI逐次轮询。不得 GPU reset 或终止其它任务。

## 观测、验收与工件

保存每组输出二进制及SHA256、guard/清零检查、CUDA错误码、装载/符号查询结果、
host launch+sync wall（不是CUDA event基准）、PID、退出状态及看护停止原因。
通过要求三组都精确等于预期字节且guard完整；无浮点容差放宽。

结果目录：/home/yjl/HBFSim/artifacts/llama2-sm80-gpu-probe-v1/execution-v1
完成/失败路径：execution-v1/DONE 或 FAILED；另有 START/PID/progress/RESULT/FINISH JSON。
脚本前台等待并写完成状态；尚未配置外部完成通知。负责代理可依预计wall/工具会话
结束检查一次，不逐kernel唤醒AI。

CPU准备已通过：参数尺寸/对齐、推导地址覆盖0..2047、无越界、三个ELF输入；
门禁检查证明 parent/worker 缺少批准时都在加载driver前拒绝。
这些不是 GPU 正确性证据。

准确启动命令（待真实确认后，由负责代理建立独立确认记录并先通过 gate）：

```bash
/var/tmp/hbfsim-yjl/llama2-layer15-env-20261019-v1/env/bin/python -B /home/yjl/HBFSim/artifacts/llama2-sm80-gpu-probe-v1/probe.py --execute --approval /home/yjl/HBFSim/artifacts/llama2-sm80-gpu-probe-v1/USER_CONFIRMATION.json
```

## 确认文本

只有实际用户发送的确认才能生效；本文件与代理生成的记录都不构成同意。
请明确确认以下绑定及上述范围：

```text
批准 experiment_id=LLAMA2-SM80-PTX-GPU-PROBE version=1 canonical_manifest_hash=a54ce76c6ee340533c953a407d126f44b257014b61481e27be8ffc797e5c297f
```

规则来源：[AGENTS.md](../../AGENTS.md) 的实验预检要求，以及
[experiment_approval.md](../../docs/eq3_thermal/experiment_approval.md) 的 ID/版本/hash 绑定契约。
