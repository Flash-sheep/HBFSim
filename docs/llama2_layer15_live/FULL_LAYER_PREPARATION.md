最新更新：CUDA11最小适配已获明确批准并通过固定CPU验证，见[兼容结果](CUDA11_COMPATIBILITY_PROGRESS.md)。以下准备期状态和旧失败保留为历史。完整GPU/HBF Timing仍未执行。

# Llama2 第15层完整验证准备（2026-10-19）

状态：CPU 输入与完整前向验证通过；完整层 GPU/HBF Timing 未执行。
CUDA11 运行库兼容方案待用户确认，未实施。

## 输入与固定软件验证

USER_CONFIRMED：用户要求“完整验证llama完整层”，并提供
`/mnt/hbfsim-workload` 和 `/mnt/llama2-7b`。
DOC_DERIVED：沿用归档层15、batch/query/context=1/1/32、FP16、seed1、warmup0。
INFERRED：先用原脚本做固定CPU检查，隔离准备依赖，不启动新的GPU负载。
这里的完整层是一个decoder layer，不是32层模型的端到端生成。

- 两份原脚本、5份小型JSON及模型配置/索引身份与归档一致；第15层10个tensor哈希一致。
  [输入身份收据](current_machine_20261019/full_layer_preparation/INPUT_IDENTITY.json)。
- 原`trace_flint_decoder_layer.py --validate-only`退出0，
  `weights=10 fp16_bytes=404766848`，未生成GPU地址manifest。
- 使用原加载器、rotary与forward调用准备函数，固定CPU执行一次完整层。
  输出FP16、`[1,1,4096]`，全部有限；9个参数的顺序、形状与256B对齐偏移
  均与原manifest一致，总计404,766,720B。
  [CPU完整前向收据](current_machine_20261019/full_layer_preparation/CPU_FULL_FORWARD.json)。
- 额外128B是旧rotary buffer，不能计为第10个注册权重。
  CPU结果不代表GPU数值等价、PTX覆盖或HBF请求/时序通过。

模型与原脚本只读使用。冻结副本和实际日志位于
`artifacts/llama2-layer15-full-validation-v1/`；大体积源码、环境与构建位于
`/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/`。

## 隔离构建准备

现有Torch wheel无嵌入PTX。已获取与其实际git版本完全相同的源码
`e4ee3be4063b7c430974252fdf7db42273388d86`；不是仅凭v2.4.0标签代替。
[框架源码身份](current_machine_20261019/full_layer_preparation/PYTORCH_SOURCE_IDENTITY.json)。
尚未构建框架或改动算子实现；后续需保留sm80 PTX并重新核对实际kernel。
原CMake架构选择函数的固定CPU检查已确认`8.0+PTX`生成sm80 cubin及compute80 PTX两组flags，
[配置检查](current_machine_20261019/full_layer_preparation/PYTORCH_ARCHITECTURE_PRECHECK.json)不代表最终库已包含PTX。

agent源为固定ec26提交、原0002补丁及20个固定子模块。
[源码准备收据](current_machine_20261019/full_layer_preparation/AGENT_SOURCE_PREPARE.json)。
跨文件系统local clone硬链接失败保留，重试使用`--no-hardlinks`，不改源码配方。
当前主机构建配置（GCC11、LLVM17、Boost1.88、CUDA12.8）已通过，
[配置收据](current_machine_20261019/full_layer_preparation/AGENT_CONFIGURE_V1.json)。
这不是原主机build lock验收通过，也没有添加CUDA11域。
首次编译因私有CUDA环境缺少NVML开发头文件失败，原失败日志与收据保留；已在私有环境补齐CUDA12.8 NVML开发头文件并重试，未改agent源码。
[只读CUDA11接口审计](current_machine_20261019/full_layer_preparation/CUDART11_INTERFACE_AUDIT.json)
确认所需9个运行库符号真实导出`libcudart.so.11.0`版本，现有gate没有该域。
第二次编译在GPU shared map翻译单元因系统旧`linux/bpf.h`缺`bpf_link_type`而失败。
构建目标没有先复制vendor的uapi头文件；第三次在新构建目录先运行仓库已有
`copy_headers`，再构建`bpftime-agent`，源码未改。此时仍在编译，未标记PASS。
本地watchdog已配置，完成通知未配置；会话、进程和DONE/FAILED路径见
[当前状态](current_machine_20261019/full_layer_preparation/PREPARATION_STATUS_V2.json)。

## 尚未闭合的部分

1. 当前gate/agent只处理CUDA12/13身份；CUDA11.7 cuBLAS虽然有sm80 PTX，
   仍需[最小兼容方案](MINIMAL_CUDART11_PLAN.md)明确确认后才能实施接口扩展。
2. 原框架sm80/PTX构建、实际kernel及容器身份、staged绑定和完整运行包装尚未闭合。
3. 完整层GPU预检需在产物冻结后提交新版本/hash确认。已批准并完成的
   [三组GPU小样本](GPU_SM80_SAMPLE_RESULT.md)不覆盖本任务。
4. 完整验收须有native数值对照、所有9个权重的addressed/active modeled覆盖、
   未覆盖字节及请求完成/错误闭合；仅注册权重或forward成功均不足。

没有启动完整层GPU任务；生产源码、CUDA原库、driver、全局环境与原raw结果未修改。
