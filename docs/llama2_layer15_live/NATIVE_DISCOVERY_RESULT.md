最新：v3严格完整层capture已实际通过；完整HBF尚未验证。见[v3结果](NATIVE_DISCOVERY_V3_RESULT.md)及[v4只读Driver观测预检](NATIVE_DISCOVERY_PREFLIGHT_V4.md)。以下保留历史过程。

最新：v2单次capture已执行，45次launch均被拒绝且输出不匹配。已完成CPU定位和隔离版本表修复；见[v2实际结果](NATIVE_DISCOVERY_V2_RESULT.md)。以下为v1历史记录。

# 2026-10-20 原生完整层GPU结果与严格接入停止记录

USER_CONFIRMED：`LLAMA2-SM80-NATIVE-DISCOVERY` version1，hash
`0da675dcfdbcf84395b09fb524335af87bfba0581620df056e218fba87b97222`。
真实用户确认已保存并经原gate检查READY；按批准命令执行一次，没有自动重试。

## 实际结果

| 环节 | 实际结果 |
| --- | --- |
| 原生完整层 | PASS，原helper，layer15、1/1/32、FP16、seed1、warmup0 |
| 权重 | 真实checkpoint的9个参数，404,766,720B，当前进程新arena与原layout一致 |
| 输出 | FP16 `[1,1,4096]`、8192B、全部有限 |
| 输出SHA256 | `528e30d35ba48d50b8aa98d8b01f4e3165a5de75522eaaa60f7170e00e6b9aea` |
| 原生host forward+sync | 0.972778秒，包含首次运行开销，不是性能基准 |
| 严格capture | 初始化阶段触发安全停止；尚未开始完整forward，没有输出可比较 |
| 总执行wall | 20.572993秒，含初始化与清理 |
| HBF注册/Timing | 未执行，不能称完整HBF验证通过 |

[原生收据](current_machine_20261019/native_discovery_v1/NATIVE_RESULT.json)，
[总失败收据](current_machine_20261019/native_discovery_v1/RESULT.json)。
原raw与二进制输出位于`artifacts/llama2-sm80-native-discovery-v1/execution-v1/`，未覆盖。
所有本任务actor已清理；后续只读检查GPU1为135MiB、0%利用率，GPU0原进程未动。

## 预载继承缺陷与证据边界

OBSERVED：capture PID3522550成功装好严格11域hook、完成空PTX bootstrap及Attach。
随后默认runtime日志又出现PID3522681初始化agent并创建CUDA context；它不在
声明的native/loader/capture主PID允许名单内。capture没有model validate、arena或
FORWARD_BEGIN。默认`~/.bpftime/runtime.log`快照已另存，原文件不改。

DOC_DERIVED：worker入口再次执行approval gate，该gate通过subprocess调用Git，
原worker没有先清理继承的LD_PRELOAD。Git及其递归子模块检查因此也会加载agent。

CPU_REPRODUCED：用仅记录PID/exe的惰性CPU标记DSO，在实际Git检查中复现24个Git
进程加载预载；v2实际清理函数执行后为0个。清理后标记DSO仍留在当前进程maps中。
这项验证没有加载CUDA库或启动GPU。四项v2 CPU安全测试也通过。
[CPU复现](current_machine_20261019/native_discovery_v1/PRELOAD_INHERITANCE_CPU.json)，
[当前映射保留](current_machine_20261019/native_discovery_v1/PRELOAD_MAPPING_RETENTION_CPU.json)。

INFERRED：嵌套Git agent与外来进程分支触发高度一致。v1把温度、GPU内存和外来PID
合并为同一错误，且没有持久化触发瞬间的sample和actor最终退出码；这些属于
UNKNOWN_NOT_CAPTURED，不能用清理后的43℃/135MiB倒推出当时的精确触发条件。
因此不能据此认定GPU或CUDA11不兼容，也不把具体停止分支写成已直接观测事实。

## 已准备的局部修复与下一步

v1批准源码、manifest与raw保留不变。新包装器`native_discovery_v2.py`只在worker
已加载agent/gate后、调用校验前，移除供子进程继承的LD_PRELOAD；当前已装载库继续工作。
同时使用任务内runtime log，保存每次GPU/resource sample、安全触发原因和actor退出码。
保持主PID允许名单及所有资源阈值，不修改生产core/ABI/PTX/调度/所有权，也没有重建原库。

成功native输出经当前4591份环境文件哈希、输出SHA和九权重layout审计，可作为同一
软件配置的字节对照复用；不是物理标定或其它设备参考。v2只补做一次capture forward，
新arena地址依然来自新进程，不复用旧指针。v2预检见
[capture-only v2](NATIVE_DISCOVERY_PREFLIGHT_V2.md)，该预检随后已批准并执行，实际结果见[v2结果](NATIVE_DISCOVERY_V2_RESULT.md)。

v1明确规定“无自动重试”；当时安全停止后未自行重启，v2随后取得独立批准后才执行。
