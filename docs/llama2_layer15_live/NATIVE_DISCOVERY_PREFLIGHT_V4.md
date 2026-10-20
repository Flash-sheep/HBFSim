# v4单次完整层Driver观测预检

状态：**PENDING_USER_APPROVAL**，没有本版本确认或GPU任务。

| 绑定 | 冻结值 |
| --- | --- |
| experiment_id | `LLAMA2-SM80-NATIVE-DISCOVERY` |
| version | `4` |
| canonical_manifest_hash | `b0cd0ed17121c225da51d54c505b03c6207a42c42c5215671f1393242421323b` |
| manifest | `artifacts/llama2-sm80-native-discovery-v4/manifest.json` |
| wrapper | `experiments/llama2_sm80_cuda11/native_discovery_v4.py` |
| 输出 | `artifacts/llama2-sm80-native-discovery-v4/execution-v1/`；已存在则拒绝 |

## 问题、方法、证据和限制

USER_CONFIRMED：原始Llama2第15层完整验证；v3单次strict capture已经通过，8192B输出
与成功native完全相同。[实际结果](NATIVE_DISCOVERY_V3_RESULT.md)同时记录了仍缺的
cuBLAS Driver观测及一个归约入口的unsupported atomic，不能称完整HBF通过。

本版本问题：可选的只读CUPTI回调能否观察到实际Driver launch，包括现有Torch入口
以外的kernel，同时保持原始完整层输出和已成功strict capture？假设仅是软件观察
路径透明，不是物理/热/可靠性假设，也不承诺额外kernel一定有可用PTX。

相对v3只新增独立CUPTI11.7观察DSO和原始记录/验收；gate、agent、pass和数学算子
不重建、不修改，不加载CPU已转换的PTX，不设置寄存器限制，不注册HBF范围或调用
Timing服务。回调无CUDA API调用，不改Driver参数、返回值、CUfunction、调度或所有权。
读取ENTER/EXIT相关ID、context、CUfunction、symbolName和result；若有
cuModuleGetFunction调用，保存其module/function/name及字符串DSO信息。
字符串DSO只是辅助观测，不自动认定它就是源PTX容器。

CPU证据：观察DSO已实际构建；六次惰性回调验证ABI、相关ID、JSON转义、未知API布局
标记、正常停止及订阅失败处理；测试未加载真实CUDA库或打开GPU设备。
包装器十项CPU正负检查通过，实际execute/worker入口均无批准拒绝。
真实CUPTI11.7与本驱动/agent组合的可用性尚未验证；任何不兼容保留为实际失败。

几何/拓扑/材料/功耗/可靠性参数、物理初态和模拟时长均为NOT_USED；这是固定、
有界软件验证。不会以性能、物理标定、完整kernel绑定或完整HBF验证PASS报告。

## 冻结代码、环境和输入

HBFSim HEAD `874a20f289b7bd587fa4048a788e12df2618444b`，tracked diff为空。
原Torch git `e4ee3be4063b7c430974252fdf7db42273388d86`，实际私有构建
2.4.0+cu117.sm80ptx，CUDA11.7、ABI0；Python3.11、Transformers4.57.1、safetensors0.6.2。
原始输入、所有原环境文件、九权重layout、十个checkpoint tensor及新观察器/实际
CUPTI/头文件哈希均绑定和重新核对。新DSO的ldd确认实际CUPTI11.7，无缺失或stub。

| 新工件 | SHA256 |
| --- | --- |
| libllama2_cupti_observer.so | `c4280f508f90571ba10bbac59fe831046b7b07a5d39fef23c7b05c3662dc7521` |
| 原libcupti.so.11.7 | `d23892b9f83727174a11b942375a7a17941198bd3b40b0da0f29e2a24abfe069` |

DOC_DERIVED：模型`/mnt/llama2-7b`、原workload`/mnt/hbfsim-workload`，layer15，
batch/query/context=1/1/32、FP16、seed1、warmup0，一次完整forward。
九个真实checkpoint参数404,766,720B，新进程生成256B对齐arena，不复用旧地址。
额外128B旧rotary buffer不是第10参数。原helper在CPU按固定seed生成hidden和
历史KV31、position31；不是32层整模型生成或真实token输入。

对照继续复用v1成功native输出（同一原始数学环境，无预载），8192B SHA256
`528e30d35ba48d50b8aa98d8b01f4e3165a5de75522eaaa60f7170e00e6b9aea`。
v3成功strict capture是观察通路先决证据，不额外重跑native或其它GPU点。

## 器件、点数、资源和安全边界

| 项目 | 执行边界 |
| --- | --- |
| 目标 | GPU1，A100-PCIE-40GB、sm80、driver565.57.01 |
| 唯一UUID | `GPU-5302a497-ea8c-bb7b-7235-f7464792093e` |
| GPU0 | 其它用户占用，排除；不改、不重置 |
| 点/重复 | 1个capture点、1次forward、warmup0；无扫描、无自动重试 |
| 辅助actor | 同GPU私有namespace server/loader一次，其context计入预算 |
| GPU时间 | 最多4分钟含清理；工作watchdog180秒，actor清理宽限各5秒 |
| 预计wall | 30–90秒，INFERRED；无CUPTI的v3实际38.297秒，不是本版本实测 |
| CPU | OMP/BLAS1、nice10，无编译；累计CPU时间上限720秒 |
| RAM | actor组RSS上限16GiB；主机可用余量保留64GiB |
| GPU内存 | 整卡已用达到12288MiB停止，包含辅助actor |
| 磁盘 | 输出上限2GiB，文件系统可用保留2GiB |
| 控制 | strict、空PTX目录、关闭cuobjdump、TF32关闭、CUDA_MODULE_LOADING=EAGER |
| cache/临时目录 | 任务内新CUDA cache/TMPDIR；清理继承的profiling环境，不改全局配置 |

启动时目标须无外来GPU进程、utilization0、内存<1024MiB、温度<75℃。
运行时任一未声明GPU PID、温度≥80℃、GPU内存≥12288MiB、RSS>16GiB、可用RAM
<64GiB、CPU时间>720秒、输出>2GiB、磁盘余量<2GiB或wall>180秒，停止自己的actor。
GPU UUID锁、父进程死亡保护、主PID白名单、内部看护、有限清理和私有SHM身份检查
保留；不添加任意子进程为GPU允许项，不改变driver/频率/功率设置。

外部完成通知未配置。保存START/PID/RESULT/DONE或FAILED；按已有wall估计在预计
完成时读取，不进行AI逐kernel轮询，内部资源安全监测继续运行。

## 观测、验收、失败和输出

无额外消融。保存原arena、FP16输出、strict coverage/runtime/bridge、11域注册、
已加载文件、CUPTI_START/STOP、callback/write-error计数、完整Driver JSONL、
forward byte窗口、实际Driver ENTER/EXIT配对、GPU/resource sample及actor退出码。
原始窗口先保存再验收；CUPTI记录不回填成已有agent的Driver bridge记录。

通过条件全部满足才报告无HBF的discovery PASS：

1. 与v3相同的数值与strict要求：一次capture成功；输出有限且8192B精确等于native；
   coverage非空且全部allowed/non-modeled；runtime有cudart11，全部成功ORIGINAL。
   零Driver调用须有现有认证original_completion；实际agent Driver调用仍须成功且有匹配bridge。
2. CUPTI订阅/停止成功、write_errors0；forward窗口有实际Driver launch。
3. launch ENTER/EXIT按context UID及correlation ID一一配对，原API/function/kernel/
   context相同且返回0；所有观察到的launch布局已受支持，无缺失function或symbol。
4. CUPTI kernel集合包含strict观察的所有Torch入口，并有至少一个额外入口；保存额外
   入口原名，不仅凭名称就认定其源容器、兼容PTX或HBF可绑定。

不满足任一CUDA/API、订阅、callback、输出、身份、strict或资源条件就停止自己的
actor，不自动重试、放宽阈值或猜测身份，原raw保留。host耗时含观察开销，不作性能结论。

已知限制：CUPTI只负责观察，不代表HBF gate已拦截内部cuBLAS路径；额外入口可能
没有唯一/可用原PTX。19个Torch入口的CPU scoped准备不等于完整GPU绑定，归约入口
的原子操作仍被拒绝。真实PTX/容器绑定、九权重active modeled及普通Timing服务
闭合属于后续工作，本版本不运行这些项目。

## 确认与准确命令

```sh
/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/torch-cu117-env/bin/python -B \
  /home/yjl/HBFSim/experiments/llama2_sm80_cuda11/native_discovery_v4.py \
  --bundle /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v4 \
  --execute \
  --approval /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v4/USER_CONFIRMATION.json
```

```text
批准 experiment_id=LLAMA2-SM80-NATIVE-DISCOVERY version=4 canonical_manifest_hash=b0cd0ed17121c225da51d54c505b03c6207a42c42c5215671f1393242421323b
```

新确认的原因：v3授权的一次capture已完成，新增CUPTI观察方法及另一次GPU前向属于
新执行版本。[AGENTS.md](../../AGENTS.md)及[实验契约](../eq3_thermal/experiment_approval.md)
要求先展示预检并取得该版本确认；此前v3批准不能覆盖本次新增负载。
