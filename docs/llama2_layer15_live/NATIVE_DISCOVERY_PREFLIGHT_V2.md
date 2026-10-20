执行后更新：v2已获批准并于2026-10-20执行一次，结果FAILED，无自动重试。见[v2实际结果](NATIVE_DISCOVERY_V2_RESULT.md)。以下保留原预检范围；原manifest和工件内预检未改。

# 严格接入发现v2：单次capture预检

状态：**PENDING_USER_APPROVAL**；没有批准记录，没有GPU提交或后台任务。

| 绑定 | 值 |
| --- | --- |
| experiment_id | `LLAMA2-SM80-NATIVE-DISCOVERY` |
| version | `2` |
| canonical_manifest_hash | `1a979d2b4624385fa10eb6f7518a0150c28062858e98e9e924b1090a950d57ba` |
| manifest | `artifacts/llama2-sm80-native-discovery-v2/manifest.json` |
| 执行源码 | `experiments/llama2_sm80_cuda11/native_discovery_v2.py` |

## 研究问题与v1差异

USER_CONFIRMED：完整验证原始Llama2第15层；v1的原生前向已成功，严格capture在初始化
时停止。[实际结果及完整证据](NATIVE_DISCOVERY_RESULT.md)保留了成功native与失败capture。

本次问题：修复worker子进程预载继承后，strict capture是否可完成原始完整层，并与
成功native输出逐字节一致？假设仅是软件观察路径透明；不涉及物理模型。

与v1相比：仅执行**一次capture**，不重跑native。复用v1已成功的8192B原生GPU输出
`528e30d35ba48d50b8aa98d8b01f4e3165a5de75522eaaa60f7170e00e6b9aea`，
不是旧CPU/其它GPU结果。4591份原环境文件哈希、原始输入和九项layout已重新核对；
原输出、RESULT和arena也加入v2身份绑定。新进程仍生成自己的arena地址。

包装器在校验前移除对子进程的LD_PRELOAD继承，当前已装载的agent/gate继续工作。
CPU实际Git复现为修复前24个Git预载、修复后0；标记库仍保留在当前进程maps中。
四项CPU安全fixture及实际两个允许GPU入口的无批准拒绝检查通过。
v1源码/manifest/raw不变；生产ABI、PTX、服务、调度和所有权逻辑不改。

新增任务内runtime log、resource sample、安全触发snapshot及actor退出码。
维持只允许主actor PID的GPU名单及全部旧阈值；不允许Git等额外进程创建CUDA context。

## 冻结环境、输入与证据身份

HBFSim HEAD `874a20f289b7bd587fa4048a788e12df2618444b`，tracked diff为空。
Torch原git `e4ee3be4063b7c430974252fdf7db42273388d86`，实际私有构建
2.4.0+cu117.sm80ptx、CUDA11.7、ABI0、239个sm80 PTX成员；原eager算子源码不替换。
Python3.11、Transformers4.57.1、safetensors0.6.2和固定agent/gate/server/loader
均与成功native及v1相同，逐文件重新核对；本次不编译或替换这些工件。

DOC_DERIVED：模型`/mnt/llama2-7b`，原workload`/mnt/hbfsim-workload`，layer15，
batch/query/context=1/1/32，FP16，seed1，warmup0，完整forward一次。
真实9参数404,766,720B；额外128B旧rotary buffer不是第10个参数。
hidden与历史KV按原固定种子合成；不是整模型32层生成或真实token流输入。

源脚本与checkpoint十个tensor身份已经重新核对。新环境与旧cu121 CPU输出不保证
相同，此处只用同一新构建的成功GPU输出作对照，不混用CPU参考。

这是有界软件工程验证；不注册HBF范围，不加载staged PTX，不请求HBF Timing服务。
所有物理器件几何/拓扑/材料/功耗/可靠性参数及模拟时长均为NOT_USED，不能将结果当物理
标定或完整HBF验证PASS。静态PTX清单不代替实际kernel及源容器/成员绑定检查。

## 器件、点数、资源与安全边界

| 字段 | 执行值 |
| --- | --- |
| GPU | GPU1，A100-PCIE-40GB，sm80，driver565.57.01 |
| 唯一UUID | `GPU-5302a497-ea8c-bb7b-7235-f7464792093e` |
| GPU0 | 其它用户占用，排除，不修改、不重置 |
| 点/重复 | 1个capture点，1次forward，warmup0；无扫描、无自动重试 |
| 辅助actor | 同GPU的私有namespace server/loader一次，可能创建自己的context，计入资源预算 |
| 环境控制 | strict；空PTX目录；cuobjdump关闭；TF32关闭；CUDA_MODULE_LOADING=EAGER |
| cache/临时文件 | 当前任务内全新CUDA cache/TMPDIR；不改全局设置 |
| CPU | OMP/BLAS1，nice10，无外部编译；累计CPU时间上限720秒（0.2 CPU-hour） |
| 主机内存 | actor进程组RSS上限16GiB；主机可用RAM保留64GiB |
| GPU内存 | 整卡已用达到12GiB停止；包含loader及capture的context |
| 磁盘 | 输出上限2GiB；所在文件系统可用低于2GiB停止 |
| GPU预算 | 最多4分钟，含初始化/清理；工作watchdog180秒，actor清理宽限各5秒 |
| 预期wall | 输入核对后20–60秒，INFERRED工程估计；尚无完成的capture实测 |

启动时GPU1须无外来进程、utilization0、已用内存<1024MiB且温度<75℃。
运行时任一未声明GPU PID、温度≥80℃、GPU已用≥12GiB、RSS>16GiB、主机RAM余量
<64GiB、CPU时间>720秒、磁盘余量<2GiB、输出>2GiB或wall>180秒，停止自己的actor。
父进程死亡时actor有kill保护；资源监控在启动器内部执行，不逐kernel唤醒AI。
外部完成通知未配置；PID、RESULT和DONE/FAILED由启动器持久化。

## 对照、观测、通过与失败

对照是身份审计通过的v1 native原始GPU输出。capture是同一原helper完整层，
使用新进程和新地址；所有变化限定在观察及启动包装，不替换数学算子。

保存九权重arena、原始FP16输出、实际GPU身份、forward起止byte offset、coverage、
11域注册、runtime/Driver严格桥接、已加载文件路径、resource-samples、触发snapshot、
actor PID/退出码及任务内bpftime runtime log。host耗时含观察/同步开销，不作性能结论。

通过：一次capture成功；输出FP16 `[1,1,4096]`且全部有限，8192B与冻结native完全一致；
forward窗口中有成功CUDA11 runtime记录和允许、非modeled的coverage；严格桥接
选择ORIGINAL、返回成功。通过状态为`NATIVE_DISCOVERY_PASS_HBF_NOT_RUN`。

任一CUDA/API、输出、身份、严格桥接或进程失败立即停止，不自动重试、不放宽数值
或安全阈值；保留raw，不补造缺失元数据。完整HBF绑定/九权重active modeled覆盖
仍要在后续实际入口及staged provenance闭合后验证。

## 输出、命令与确认

输出：`artifacts/llama2-sm80-native-discovery-v2/execution-v1/`，目录已存在则拒绝运行。
当前没有创建该目录或USER_CONFIRMATION记录。

确认后执行：

```sh
/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/torch-cu117-env/bin/python -B \
  /home/yjl/HBFSim/experiments/llama2_sm80_cuda11/native_discovery_v2.py \
  --bundle /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v2 \
  --execute \
  --approval /home/yjl/HBFSim/artifacts/llama2-sm80-native-discovery-v2/USER_CONFIRMATION.json
```

请确认具体版本：

```text
批准 experiment_id=LLAMA2-SM80-NATIVE-DISCOVERY version=2 canonical_manifest_hash=1a979d2b4624385fa10eb6f7518a0150c28062858e98e9e924b1090a950d57ba
```

需重新确认的原因：v1预检明确“无自动重试”，本次实际执行已安全停止；此处是带已验证
局部包装修复、复用既有native且只补一次capture的新版本。[AGENTS.md](../../AGENTS.md)
及[实验确认契约](../eq3_thermal/experiment_approval.md)要求此新GPU执行版本先确认。
