最新：v3严格完整层capture已实际通过；完整HBF尚未验证。见[v3结果](NATIVE_DISCOVERY_V3_RESULT.md)及[v4只读Driver观测预检](NATIVE_DISCOVERY_PREFLIGHT_V4.md)。以下保留历史过程。

# 2026-10-20 v2单次capture实际结果

USER_CONFIRMED：`LLAMA2-SM80-NATIVE-DISCOVERY` version2，hash
`1a979d2b4624385fa10eb6f7518a0150c28062858e98e9e924b1090a950d57ba`。
实际用户原文、确认记录和READY gate已保存。只执行一次capture，没有重跑native或自动重试。

| 检查 | OBSERVED结果 |
| --- | --- |
| 总状态 | FAILED，worker退出2，loader清理退出0，总wall35.119362秒 |
| 原始helper | layer15、batch/query/context=1/1/32、FP16、seed1、warmup0、一次forward及同步返回 |
| 输出 | `[1,1,4096]`、8192B、全部有限；**与成功native不一致** |
| 数值差异 | 4096/4096元素不同，8167/8192字节不同，最大绝对差3.564453125 |
| capture SHA256 | `9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47` |
| 原生对照SHA256 | `528e30d35ba48d50b8aa98d8b01f4e3165a5de75522eaaa60f7170e00e6b9aea` |
| forward窗口runtime | 45条，全部unversioned、返回801、`REJECTED_DRIVER_CORRELATION`、driver_calls0 |
| coverage / bridge | 均为0；worker断言`No actual forward coverage/strict bridge records` |
| 注册身份 | 239个CUDA11 fatbin、16798个函数注册；45次尝试涉及20个host stub，均能精确关联注册记录 |
| 资源采样 | 28份sample；采样峰值GPU2165MiB、43℃、组RSS3179061248B；无未声明GPU PID |
| 结束 | 所有本任务actor退出，后续只读检查GPU1为135MiB/0%/43℃；GPU0未改 |
| HBF范围/Timing | 均未执行，完整HBF验证未通过 |

没有`SAFETY_STOP.json`；这次是worker验收失败，不是v1的资源/外来进程安全停止。
forward host区间0.957083秒包含观察开销且计算输出错误，不能当作正确性或性能依据。

[总结果](current_machine_20261019/native_discovery_v2/RESULT.json)、
[完整分析](current_machine_20261019/native_discovery_v2/RESULT_ANALYSIS.json)、
[raw文件哈希清单](current_machine_20261019/native_discovery_v2/RAW_INVENTORY.json)。
原始日志、输出、arena、PID及resource sample完整保存在
`artifacts/llama2-sm80-native-discovery-v2/execution-v1/`，未覆盖或修改。

## CPU定位与隔离修复

CPU_REPRODUCED：生成真实ELF消费者，链接具有`libcudart.so.11.0`版本的launch导入；
预载冻结实际gate，但用惰性CUDA ABI替身替代全部CUDA provider。消费者实际重定位
选择gate未版本化的`cudaLaunchKernel`拒绝入口，而非通过`dlvsym`可见的精确11入口；
返回3，fake11 provider调用0次。无agent构造、真实CUDA provider、设备访问或GPU任务。

DOC_DERIVED：实验gate的未版本化入口只返回初始化错误；agent generic hook没有11域
认证握手，因没有gate/Driver关联而返回801。这与实际45条unversioned拒绝记录一致。
数值错误可能涉及初始化和forward，尚未定位到逐个算子，不能断言只是输出校验器问题。

在此前已批准的“独立副本CUDA11版本符号解析和对应launch入口”范围内，仅给实验ON
版本表中的拒绝入口分配独立版本节点，避免遮蔽显式11版本。用冻结原对象独立重链接
新库；原gate/源码/库/manifest不变。HBF导出、显式11/12/13别名、函数实现及默认OFF
构建均不变。没有新增名称猜测、弱身份匹配或放行条件。

修复后相同CPU消费者选择精确11入口，返回0且只调用fake11一次。12个legacy、八个
CUDA11检查，以及实际agent注册、未就绪launch拒绝和错误身份拒绝通过。
[修复前](current_machine_20261019/native_discovery_v2/ELF_LOOKUP_BEFORE_CPU.json)、
[修复后](current_machine_20261019/native_discovery_v2/ELF_LOOKUP_AFTER_CPU.json)、
[独立链接与身份](current_machine_20261019/native_discovery_v2/ELF_FIX_BUILD.json)、
[CPU回归](current_machine_20261019/native_discovery_v2/ELF_FIX_CPU_REGRESSION.json)。

新库位于`/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/cudart11-launch-lookup-v2/`。
其SHA256为`1dc9125011fc97a8195cf289a4b6c4942fff77505e5b4341680888c675b4f19b`。
首次CPU准备把linker的soname选项误当文件，尚未链接就失败；该失败目录单独保留，
修正文件清点后才在新attempt中链接成功，不修改或冒充旧失败收据。

## 剩余验收与下一步

DOC_DERIVED：现有strict ORIGINAL路径支持gate认证的一次性original-completion握手，
此时不一定有Driver拦截记录。v2把非空Driver bridge作为无条件要求，未完整表达该
已有契约。v3要求成功coverage和runtime，零Driver调用必须有真实认证握手；有Driver
调用则仍要求成功且匹配的bridge。八项CPU正负fixture通过，不伪造Driver记录。

这些CPU结果不证明GPU修复成功。239个静态PTX及20个尝试入口也不证明实际成功执行
的kernel集合、精确PTX成员绑定、cuBLAS内部launch覆盖或九权重HBF服务闭合。
完整HBF仍需后续准备和独立批准。下一次单capture GPU复验见
[v3预检](NATIVE_DISCOVERY_PREFLIGHT_V3.md)，目前PENDING_USER_APPROVAL，未启动。
