执行后更新：v3已获批准并执行一次，结果NATIVE_DISCOVERY_PASS_HBF_NOT_RUN，输出与native逐字节一致。见[v3实际结果](NATIVE_DISCOVERY_V3_RESULT.md)。以下为原预检；工件内原预检及manifest未改。

# 严格接入发现v3：ELF版本修复后的单次capture预检

状态：**PENDING_USER_APPROVAL**；没有v3确认记录、GPU提交或后台任务。

| 绑定 | 值 |
| --- | --- |
| experiment_id | `LLAMA2-SM80-NATIVE-DISCOVERY` |
| version | `3` |
| canonical_manifest_hash | `b179eb21beaf9bb8a24ebf06f5f19f7953d22e52d6ea76cadcd6463460ea116f` |
| manifest | `artifacts/llama2-sm80-native-discovery-v3/manifest.json` |
| 执行源码 | `experiments/llama2_sm80_cuda11/native_discovery_v3.py` |
| 输出 | `artifacts/llama2-sm80-native-discovery-v3/execution-v1/`；已存在则拒绝 |

## 问题、证据身份与变更

USER_CONFIRMED：完整验证原始Llama2第15层。v1 native成功；v2单capture已获批准并
执行一次，但45次launch均被拒绝且输出不匹配。没有自动重试。
[实际结果、CPU复现和独立修复](NATIVE_DISCOVERY_V2_RESULT.md)保留了全部失败。

本次研究问题：实验ON版本表修复后，strict capture能否成功运行原始完整层，并保持
与已成功native完全相同的FP16输出？假设仅为软件观察路径透明，不涉及物理模型。

CPU_REPRODUCED：冻结gate的未版本化拒绝入口遮蔽显式CUDA11版本重定位。新实验
版本表将这个拒绝入口赋予独立版本；真实ELF消费者现在进入精确11 gate，只调用
惰性11 provider。12个legacy、八个CUDA11用例、实际agent注册/拒绝检查通过。
新gate SHA256：`1dc9125011fc97a8195cf289a4b6c4942fff77505e5b4341680888c675b4f19b`。
仅重链接原冻结对象，原函数实现、显式11/12/13别名和HBF导出不变；原库及默认OFF
构建未改。这是已批准的隔离CUDA11接口扩展内修复，不修改CUDA库、生产ABI或PTX。

DOC_DERIVED：现有gate/agent ORIGINAL路径允许一次性认证的original-completion，
不要求必有Driver拦截。v3验收按这一现有契约区分零/一次Driver调用；有实际Driver
调用则仍要求成功且匹配的bridge。任何拒绝、未认证、错误选择或输出不匹配都失败。
八项CPU包装器及验收正负测试通过。实际两个GPU入口均已验证无批准拒绝，不创建
execution目录。CPU验证不能替代GPU结果，也不证明完整HBF接入成功。

## 冻结代码、环境、输入与初始状态

HBFSim HEAD `874a20f289b7bd587fa4048a788e12df2618444b`，tracked diff为空。
Torch原git `e4ee3be4063b7c430974252fdf7db42273388d86`，私有构建
2.4.0+cu117.sm80ptx、CUDA11.7、ABI0、239个sm80 PTX成员；Python3.11、
Transformers4.57.1、safetensors0.6.2。agent/server/plugin/loader和原数学算子不替换。
原环境、原始输入、九权重layout、十个checkpoint tensor、新gate及链接输入均重新
核对并绑定。新gate实际ldd依赖闭合，无缺失库、无CUDA stub误加载。

DOC_DERIVED：模型`/mnt/llama2-7b`，原workload`/mnt/hbfsim-workload`。
layer15，batch/query/context=1/1/32，FP16，seed1，warmup0，完整forward一次。
原helper从checkpoint加载9参数404,766,720B，新进程生成256B对齐arena，旧指针
不复用。额外128B旧rotary buffer不是第10个参数。hidden/KV按原固定种子在CPU
生成，历史KV31、position31；这是原单层forward，不是32层整模型生成或真实token流。

对照复用v1 native成功GPU输出8192B，SHA256
`528e30d35ba48d50b8aa98d8b01f4e3165a5de75522eaaa60f7170e00e6b9aea`。
native没有预载gate，新gate链接修复不改变该原生数学环境。对照输出、结果和arena
已单独核对；不用失败v2输出或旧CPU/cu121环境替代，不再执行一次native。

本次不注册HBF范围、不加载staged PTX、不请求HBF Timing。器件/拓扑/几何/材料/
功耗/可靠性参数、模拟时长均为NOT_USED；这是有界软件工程验证，不是物理标定。
不复用旧Qwen、其它机器或旧trace55的kernel成员清单作为当前动态覆盖。

## 点数、资源与停止条件

| 字段 | 冻结值 |
| --- | --- |
| 唯一GPU | GPU1，A100-PCIE-40GB，sm80，driver565.57.01 |
| UUID | `GPU-5302a497-ea8c-bb7b-7235-f7464792093e` |
| GPU0 | 其它用户占用，排除；不修改、不重置 |
| 点数/重复 | 一个capture点、一个forward、warmup0；无扫描、无自动重试 |
| 辅助actor | 同一GPU私有namespace server/loader一次，其context计入预算 |
| GPU预算 | 最多4分钟含清理；工作watchdog180秒，actor清理宽限各5秒 |
| 预期wall | 20–60秒，INFERRED工程估计；v2实际35.119秒但失败，尚无成功capture实测 |
| CPU | OMP/BLAS1，nice10，无编译；累计CPU时间上限720秒 |
| 主机内存 | actor组RSS上限16GiB，主机可用RAM保留64GiB |
| GPU内存 | 整卡已用达到12GiB停止，包含loader及capture |
| 磁盘 | 输出上限2GiB，文件系统可用保留2GiB |
| 环境 | strict、空PTX目录、cuobjdump关闭、TF32关闭、CUDA_MODULE_LOADING=EAGER |
| cache/临时文件 | 当前任务内新CUDA cache和TMPDIR，不改全局设置 |

启动时GPU1须无外来进程、utilization0、内存<1024MiB且温度<75℃。
运行时任一未声明GPU PID、温度≥80℃、GPU已用≥12288MiB、RSS>16GiB、主机余量
<64GiB、CPU时间>720秒、输出>2GiB、磁盘余量<2GiB或wall>180秒，即停止自己的
actor。保留主PID允许名单，不把Git/任意子进程加入GPU白名单。

启动器持有GPU UUID锁，actor有父进程死亡保护，有限SIGTERM/KILL清理与私有SHM
身份核对。内部看护保留，AI不逐kernel检查。外部完成通知没有配置；会话/PID及
START、RESULT和DONE/FAILED由启动器保存。预计完成后读取结果。

## 观测、对照、通过与失败

同一新构建的v1 native是软件对照；没有额外消融或其它点。
保存九权重新地址、原始FP16输出、forward开始/结束日志byte offset、coverage、
runtime/bridge、11域注册、已加载文件、任务内bpftime log、resource sample、
触发snapshot和actor退出码。forward原始窗口先保存再验收，失败也保留。
host耗时含观察/同步/首次调用开销，不作为性能结论。

全部通过才报告`NATIVE_DISCOVERY_PASS_HBF_NOT_RUN`：

1. 一次capture进程成功，FP16 `[1,1,4096]`全部有限，8192B与冻结native逐字节相同。
2. forward coverage非空，所有记录allowed且非modeled、不要求instrumented execution。
3. runtime非空且包含cudart11；全部返回0、outcome ORIGINAL、gate_decision1、无patched alias。
4. runtime的driver_calls0必须有现有gate一次性握手认证的original_completion及gate_runtime_result0。
5. driver_calls1必须driver_result0、原/选function非空且相同，并有匹配的成功bridge；其它计数失败。
6. 所有实际bridge记录均必须选择原function、ORIGINAL、decision1、cuda_result0。

这与v2不同：Driver log不再作为无条件非空项；无Driver路径由已有认证握手证明，
不能用“没记录所以通过”。CPU正负测试覆盖伪造/缺失握手、拒绝结果、缺少coverage、
错误域、错误Driver选择等。数值比较和安全阈值没有放宽。

任一CUDA/API、输出、身份、观察路径或actor失败即停止，没有自动重试、阈值放宽
或fallback。即便通过，也只证明本次无HBF范围的观察路径；实际PTX/container绑定、
cuBLAS内部launch覆盖、九权重active modeled访问及普通Timing服务闭合仍须后续验证。

## 执行命令与确认

```sh
/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/torch-cu117-env/bin/python -B \
  /home/yjl/HBFSim/experiments/llama2_sm80_cuda11/native_discovery_v3.py \
  --bundle /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v3 \
  --execute \
  --approval /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v3/USER_CONFIRMATION.json
```

确认具体执行版：

```text
批准 experiment_id=LLAMA2-SM80-NATIVE-DISCOVERY version=3 canonical_manifest_hash=b179eb21beaf9bb8a24ebf06f5f19f7953d22e52d6ea76cadcd6463460ea116f
```

需新确认的原因：v2预检明确“无自动重试”，且已经失败；本次使用新的隔离gate和
显式更新的观察验收方法。[AGENTS.md](../../AGENTS.md)要求“在任何……新增GPU持续
负载之前……取得用户对该版本的明确确认”，详见[实验契约](../eq3_thermal/experiment_approval.md)。
v2批准不能替代本版本批准。目前没有v3的USER_CONFIRMATION或execution目录。
