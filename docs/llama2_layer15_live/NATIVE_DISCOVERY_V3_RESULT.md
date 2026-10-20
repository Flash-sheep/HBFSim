# 2026-10-20 v3完整层严格capture通过

USER_CONFIRMED：`LLAMA2-SM80-NATIVE-DISCOVERY` version3，hash
`b179eb21beaf9bb8a24ebf06f5f19f7953d22e52d6ea76cadcd6463460ea116f`。
确认原文与READY gate保存后执行一次，没有重跑native或自动重试。

| 检查 | OBSERVED结果 |
| --- | --- |
| 总状态 | `NATIVE_DISCOVERY_PASS_HBF_NOT_RUN` |
| 原始完整层 | layer15、batch/query/context=1/1/32、FP16、seed1、warmup0、一次forward |
| 数值对照 | FP16 `[1,1,4096]`、8192B、全部有限，与v1成功native逐字节相同 |
| 输出SHA256 | `528e30d35ba48d50b8aa98d8b01f4e3165a5de75522eaaa60f7170e00e6b9aea` |
| 严格runtime | 45条，全部cudart11、ORIGINAL、result0、gate_decision1、认证original_completion |
| coverage | 45条，全部allowed且非modeled，不要求instrumented execution |
| Driver bridge | 0条；45个runtime记录均driver_calls0，符合原有认证ORIGINAL握手契约 |
| 可见kernel | 20个不同的Torch入口，全部能关联当前进程的CUDA11注册身份 |
| host forward+sync | 0.995149秒，包含首次调用/观察开销，不是性能基准 |
| 总wall | 38.296814秒，含身份检查、初始化及清理 |
| 进程 | loader PID3544364、capture PID3544413均退出0 |
| 资源采样 | 30份；GPU峰值2139MiB、43℃、组RSS3202076672B；无未声明GPU PID |
| 结束 | GPU1恢复135MiB/0%/43℃；GPU0未改 |
| HBF范围、staged PTX、Timing | 均未执行；**完整HBF验证仍未通过** |

[总结果](current_machine_20261019/native_discovery_v3/RESULT.json)、
[capture结果](current_machine_20261019/native_discovery_v3/CAPTURE_RESULT.json)、
[完整分析](current_machine_20261019/native_discovery_v3/RESULT_ANALYSIS.json)、
[raw哈希清单](current_machine_20261019/native_discovery_v3/RAW_INVENTORY.json)。
原始日志、输出、arena、PID和resource sample在
`artifacts/llama2-sm80-native-discovery-v3/execution-v1/`，未修改。
v1、v2失败记录保留。生产源码、原CUDA库、driver和全局环境未改。

## 已完成的独立CPU准备

20个可见Torch入口唯一对应原始`libtorch_cuda.so`的12个PTX成员，均为原sm80/PTX7.7。
运行库注册对象所在DSO、原始容器SHA、唯一entry membership和实际launch符号已关联。
[源身份关联](current_machine_20261019/native_discovery_v3/OBSERVED_SOURCE_JOIN.json)。
这不表示原native cubin就是PTX JIT，也不表示已经绑定instrumented CUfunction。

按这一固定软件集合做CPU检查，原始成员不改：先使用现有pass转换，再在新副本中
仅将`.version7.7`适配到8.7，保持sm80和原module identity，保留所有初次失败。
初轮两个模块因有调用而停止；证明调用仅为参数来自静态模块诊断字符串的
`__assertfail`后继续。新证明不假定断言分支不可达，不改断言或PTX实现。
另两个模块因helper寄存器134超过入口限制128而失败；用agent已有
`HBFSIM_NVPTX_MAX_REGISTERS`对应的`--maxrregcount=64`选项在独立attempt中检查通过，
没有改helper或原PTX。这里只是CPU编译选项证据，尚未用于GPU任务。

结果：**19个入口、11个成员**的转换和源/派生版本身份关联通过现有scoped join，
状态`SCOPED_READY`且whole_module_ready为false。其它入口继续严格拒绝。
选定入口可生成sm80 relocatable object；这不是完整模块运行时链接、装载或GPU数值证明。

一个归约入口仍明确失败：`reduce_kernel<...MeanOps...>`中的`atom.global.add.u32`
被pass标为unsupported，instrumented=false；没有把它改为PASS或忽略该原子操作。
它可能只读activation并操作辅助计数器，但这属于INFERRED，后续必须证明地址与HBF
注册范围不相交，或另行审核受支持方案；不能靠kernel名称放行。

CPU工件在`/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/`下的
`observed-torch-stage-v1/`、`observed-torch-finish-v1/`、`observed-torch-scoped-v1/`。
[初轮检查](current_machine_20261019/native_discovery_v3/TORCH_STAGE_INITIAL.json)、
[局部复验](current_machine_20261019/native_discovery_v3/TORCH_STAGE_FOLLOWUP.json)、
[scoped结果](current_machine_20261019/native_discovery_v3/TORCH_SCOPED_STAGE.json)。

## 完整HBF尚缺的证据

OBSERVED：cuBLAS和cuBLASLt 11.7实际已加载，但目前45条coverage的kernel全部来自
Torch，未获得cuBLAS内部Driver launch记录。不能把这些记录当作全部kernel集合，
也不能以数值正确推导七个线性权重已被HBF建模。

下一步使用独立、可选、无构造器的只读CUPTI记录器补全实际Driver观测。
回调仅复制Driver参数/符号/相关ID和返回值，不调用CUDA API、不改参数/返回值。
编译和六次惰性CUPTI回调fixture通过，真实CUDA库及GPU均未被测试fixture加载；
v4包装器十项CPU检查及两个实际无批准拒绝入口通过。
[CUPTI CPU收据](current_machine_20261019/native_discovery_v3/CUPTI_OBSERVER_CPU.json)。
初次构建因重复包含无include-guard的生成头失败，错误输出和失败attempt保留；
去掉重复包含后在新目录构建成功。

当前没有新GPU授权。具体单次复验见[v4预检](NATIVE_DISCOVERY_PREFLIGHT_V4.md)。
即使得到额外kernel清单，仍需真实容器/entry/PTX绑定、九权重addressed/active modeled
覆盖及普通HBF Timing请求完成闭合；不使用fixed-delay替代服务证据。
