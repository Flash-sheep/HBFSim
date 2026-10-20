# Llama2 完整层原生对照与严格接入发现：GPU预检v1

2026-10-20更新：此v1已获用户按hash批准并执行；native完整层PASS，capture初始化阶段
安全停止。原预检与raw保留，见[实际结果](NATIVE_DISCOVERY_RESULT.md)。
仅补做capture的[v2预检](NATIVE_DISCOVERY_PREFLIGHT_V2.md)另处于PENDING_USER_APPROVAL。
下文为当时提交确认的v1原文。

状态：**PENDING_USER_APPROVAL**。当前未提交、未后台启动本次GPU任务，未创建批准记录。

- experiment_id：`LLAMA2-SM80-NATIVE-DISCOVERY`
- version：`1`
- canonical_manifest_hash：`0da675dcfdbcf84395b09fb524335af87bfba0581620df056e218fba87b97222`
- 机器可读manifest：`artifacts/llama2-sm80-native-discovery-v1/manifest.json`
- 启动代码：`experiments/llama2_sm80_cuda11/native_discovery.py`

## 问题、证据及范围

USER_CONFIRMED：完整验证原始Llama2 decoder layer；输入目录为
`/mnt/hbfsim-workload`、`/mnt/llama2-7b`。此前批准的最小CUDA11方案仅覆盖隔离实现
与固定CPU验证，旧三组GPU小样本也不覆盖本次实际Torch运行。

DOC_DERIVED：沿用原脚本layer index15、batch/query/context=1/1/32、FP16、seed1、
warmup0。这是一个完整decoder layer，不是32层整模型生成；权重真实，hidden与历史
KV由原脚本固定种子生成。9个参数共404,766,720B；第10条旧rotary buffer另128B。

问题：新构建是否能执行该完整层，且实际CUDA11严格桥接观察能保持相同输出？
假设：原生forward成功；没有HBF范围时严格桥接仍选择原始函数，两次输出逐字节一致。
本次只提供软件工程证据，不是性能基准、物理标定或HBF Timing通过。
不注册HBF范围，不加载staged PTX，不以未绑定的原生运行冒充HBF服务。

静态PTX清单无法知道运行时选择的kernel。需要先保存真实入口、注册及Driver记录，
才能继续审查源容器/成员、staged身份和HBF绑定。未知项保持UNKNOWN，不猜测入口。

## 代码、环境与既有验证

| 项目 | 冻结值与来源 |
| --- | --- |
| HBFSim HEAD | `874a20f289b7bd587fa4048a788e12df2618444b`，tracked diff为空 |
| 原Torch源码 | `e4ee3be4063b7c430974252fdf7db42273388d86`，72个递归gitlink闭合 |
| 实际新环境 | Python3.11、Torch2.4.0+cu117.sm80ptx、Transformers4.57.1、safetensors0.6.2 |
| Torch构建 | CUDA11.7、sm80 cubin+compute80 PTX、C++ ABI0；实际239个sm80 PTX成员 |
| CPU可选后端 | OpenBLAS；原eager算子源码不替换；未启用cuDNN/fused attention/分布式 |
| agent | 固定ec26+既有0002补丁+已批准的独立默认OFF CUDA11适配及冷启动provider修复 |
| HBF/agent构建工具 | 私有CUDA12.8、GCC11、LLVM17、Boost1.88、固定私有libbpf |
| driver | 565.57.01，真实driver文件哈希绑定 |
| 依赖冻结 | 4591份代码/配置/DSO及ldd依赖文件；每次spawn前重新检查 |
| 输入身份 | 原脚本、配置/索引和10个checkpoint tensor重新逐字节核对 |

独立gate/agent及剩余runtime工件均构建成功。新增11域、原12/13回归、注册与冷启动
CPU检查通过；本次agent/gate二进制与已测版本哈希一致。系统环境、原库和生产源码未改。

新环境原helper完整层CPU前向PASS。它与旧cu121 wheel的CPU输出有597/4096个FP16元素
不同，最大绝对差0.001953125；均有限，具体差异算子未定位。两套CPU后端配置不同，
不声称跨环境字节等价；本次对照严格使用同一新环境，不能用旧CPU输出充当GPU基准。

启动器三项固定CPU安全fixture及实际三个GPU入口的无批准拒绝检查已通过。
这些检查均未初始化CUDA、未启动GPU负载；实际GPU成功仍未知。

## 器件、控制与资源

| 字段 | 值与身份 |
| --- | --- |
| 目标GPU | GPU1，NVIDIA A100-PCIE-40GB，sm80；本机只读观测 |
| 唯一UUID | `GPU-5302a497-ea8c-bb7b-7235-f7464792093e` |
| GPU0 | 其它用户PID2887028占用，明确排除 |
| 拓扑/几何/材料/物理功耗/可靠性 | NOT_USED；本次不运行封装/热/HBF模型 |
| 功率/频率/driver配置变更 | 无 |
| HBF profile与模拟时长 | NOT_USED；没有HBF请求或模拟时间 |
| CPU | OMP/OpenBLAS/MKL=1，nice10；无外部编译；累计观测CPU时间上限720秒 |
| 内存 | actor进程组RSS上限16GiB，主机可用RAM保留64GiB |
| GPU内存 | 整卡已用达到12GiB停止；包括本任务loader context和目标进程 |
| 输出预算 | 最多2GiB；输出所在文件系统可用低于2GiB停止 |
| GPU预算 | 最多4分钟，包含启动与清理；工作watchdog180秒 |
| 预期wall | 输入核对后约30–90秒，INFERRED工程估计，尚无本次完整GPU实测 |

两个arm均用全新进程、原helper新arena地址与256B对齐，不复用旧地址。
TF32关闭；CUDA_MODULE_LOADING=EAGER为明确软件控制，不是物理假设。
每arm使用任务目录内独立新CUDA cache，TMPDIR也独立；不改全局环境。

## 点、对照、顺序与指标

1. **native**：不预载agent/gate，原始完整层执行一次，warmup0。
2. **capture**：同一新环境、同一原始输入，原始完整层执行一次，warmup0；
   预载实际已测CUDA11 agent/gate，strict策略，PTX目录为空。

共2次完整forward，无扫描、无自动重试。第二arm之前启动一次私有namespace的
server/attach loader；loader可能初始化同一目标GPU的context，也属于本预算。
loader与目标进程均只看见指定UUID。私有SHM名称、inode、UID单独核对，
只清理本任务的actor和匹配SHM，不删除其它bpftime对象。

观测：实际UUID/sm、温度/内存/占用、PID/退出码、arena九项地址与layout、
forward起止日志byte offset、coverage、CUDA11注册、runtime/Driver严格桥接记录、
已装载文件路径，以及每arm输出8192B FP16二进制与SHA256。
forward窗口排除初始化和尾部正确性检查。host耗时包含观察和同步开销，不作性能结论。

## 通过、失败与安全停止

通过条件：两个单次forward成功，输出均为FP16 `[1,1,4096]`且有限、8192B逐字节一致；
capture中有实际forward coverage/严格桥接及成功的CUDA11 runtime记录。
所有观测coverage应允许且非modeled；桥接必须选择ORIGINAL并返回成功。
通过状态明确为`NATIVE_DISCOVERY_PASS_HBF_NOT_RUN`，不能报告完整HBF验证通过。

任一CUDA/API、shape/输出、身份、桥接、子进程失败即停止后续actor，保留raw，不放宽阈值。
原生路径也可能因本机尚未验证的strict接口而失败，失败记录是有效排查证据。

安全门槛：启动时GPU1须无外来进程、utilization0、已用内存<1024MiB、温度<75℃；
运行时发现外来进程、温度≥80℃、GPU内存≥12GiB、RSS>16GiB、RAM余量<64GiB、
CPU时间>720秒、输出>2GiB、磁盘余量<2GiB或工作wall>180秒，立即停止自己actor。
每actor终止宽限5秒后强制结束；目标进程有parent-death kill保护。
资源监控由启动器内部执行，不逐kernel唤醒AI；未配置外部完成通知。

## 位置、命令与确认

输出：`/home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v1/execution-v1/`。
包含START、各actor PID、分arm原始日志/输出、RESULT、DONE或FAILED。
目录已存在时拒绝重跑，不覆盖raw。

确认后唯一执行命令：

```sh
/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/torch-cu117-env/bin/python -B \
  /home/yjl/HBFSim/experiments/llama2_sm80_cuda11/native_discovery.py \
  --bundle /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v1 \
  --execute \
  --approval /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v1/USER_CONFIRMATION.json
```

请明确确认这一版本，可复制：

```text
批准 experiment_id=LLAMA2-SM80-NATIVE-DISCOVERY version=1 canonical_manifest_hash=0da675dcfdbcf84395b09fb524335af87bfba0581620df056e218fba87b97222
```

依据[AGENTS.md](../../AGENTS.md)和[实验确认契约](../eq3_thermal/experiment_approval.md)，
新增GPU负载需要本版本确认。“继续”、历史最小适配批准和旧GPU小样本批准均不能
替代本次确认。不会由代理自行制造用户批准，未确认时不启动。
