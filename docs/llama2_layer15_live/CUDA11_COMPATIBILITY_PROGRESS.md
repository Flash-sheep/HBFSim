# 已批准的CUDA11隔离适配进展

最新GPU结果：v3严格capture完整原始第15层已PASS，8192B FP16输出与成功native逐字节一致，45条CUDA11 ORIGINAL认证记录全部成功；完整HBF仍未验证。见[v3实际结果](NATIVE_DISCOVERY_V3_RESULT.md)。
CPU源身份关联覆盖20个Torch入口/12个原sm80 PTX成员，19个入口/11成员完成scoped准备；归约入口的unsupported atomic保留。cuBLAS内部Driver观测仍缺。
独立只读CUPTI观察器及v4包装器已完成CPU验证，下一次单forward的[v4预检](NATIVE_DISCOVERY_PREFLIGHT_V4.md)PENDING_USER_APPROVAL，未启动GPU。原v1/v2失败及v3 raw均保留。

## 2026-10-20 实际续验结果

Torch精确源码构建已完成，wall5337.386秒（含一次158秒协调暂停），私有wheel
`torch-2.4.0+cu117.sm80ptx`安装与pip检查通过；CUDA11.7、ABI0、原git身份一致。
原helper完整层CPU前向再次PASS，输入和十个checkpoint tensor重新核对；CUDA未初始化。
实际`libtorch_cuda.so`已提取239个sm80 PTX成员，不只是NVCC配置flags。
[构建收据](current_machine_20261019/cuda11_compat/framework-build-v1.json)，
[新环境CPU前向](current_machine_20261019/cuda11_compat/framework-CPU_FULL_FORWARD_CU117.json)。

旧cu121 wheel与新构建的CPU输出不逐字节一致：597/4096元素不同，最大绝对差
0.001953125，均有限。两者CPU可选后端配置不同，具体差异算子未定位，不能声称
跨环境数值等价。后续GPU对照必须使用同一新构建；保留两份原输出和差异收据。
[CPU差异](current_machine_20261019/cuda11_compat/framework-CPU_BUILD_COMPARISON_V1.json)。

原CPU后处理FAILED保留：缺少libbpf的pkg-config配置，CMake未生成attach loader目标。
已仅在任务目录提供指向固定agent私有libbpf的`.pc`，重新配置后剩余HBF和server工件
全部构建通过（56.500秒）。未安装系统依赖；实际agent/gate二进制哈希仍与已测CPU版本一致。
[原失败](current_machine_20261019/cuda11_compat/framework-postbuild-v1.json)，
[恢复构建](current_machine_20261019/cuda11_compat/framework-finish-build-v1.json)。

此前CPU准备时，原生完整层对照与严格接入发现的GPU预检状态为PENDING_USER_APPROVAL；
它只运行两个单次原生完整前向，不注册HBF范围，不加载插桩PTX，不是HBF Timing验证。
启动器和两种worker入口实际无批准拒绝检查通过；私有运行库/依赖与4591份文件已冻结。
预检包为`artifacts/llama2-sm80-native-discovery-v1/`，说明见
[GPU原生发现预检](NATIVE_DISCOVERY_PREFLIGHT.md)。该历史准备阶段GPU任务数为0，后续实际执行见本文顶部。

下文为此前适配与构建过程记录；其中“构建仍在运行”等状态已由上述实际收据替代。

USER_CONFIRMED：用户在最小方案链接后回复“批准”。原方案副本SHA256为
`7ceb358d6e3e64a0642a630fb3efb55363d16c850ae593f8d4b61228e6d2df99`。
真实确认、逐字用户来源和原方案副本保存在
`artifacts/llama2-layer15-full-validation-v1/cuda11-plan-approval-v1/`。
这项批准覆盖隔离兼容实现和固定CPU验证，不覆盖GPU启动。

代码包：[experiments/llama2_sm80_cuda11](../../experiments/llama2_sm80_cuda11/)。
生产源码和原CUDA库未改。gate/agent补丁仅应用在
`/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/cudart11-v1/`源码副本。
`HBFSIM_EXPERIMENTAL_CUDART11`默认OFF；ON仅支持CUDA11的严格public host-stub
kernel launch。PTSZ/cooperative入口显式拒绝，不伪造CUDA11没有的私有kernel入口。
保留原注册、generation、严格返回凭据和native binding校验；不改生命周期或所有权。

原agent本机构建已PASS（359.289秒），不是原主机build lock的PASS。
隔离gate OFF/ON构建均PASS，CUDA12/13原12项CPU检查在两种gate上均PASS；
ON时只将原负例中的无效域11改成14，其余12/13检查保留。
新增8组CUDA11 CPU检查PASS：真实版本分流、0/801返回、错误域/函数、跨线程、
三种未审查API拒绝。另核对默认OFF没有11版本launch导出，ON没有私有11导出，
非strict拒绝。检查中libcuda完全由CPU fake DSO替代，无GPU设备使用。

agent OFF/ON均构建PASS。实际注册监听器CPU fixture已PASS并正常退出。
固定注册fixture已覆盖11/12/13相同数值句柄隔离、真实Gum回调的注销/复用、
合成stale generation故障，以及native binding错误container/domain/entry拒绝。
还覆盖实际agent/Gum→gate的未就绪和非strict CUDA11 launch拒绝。
首次registry fixture退出double free保留为FAILED；回溯定位为测试链接去掉版本脚本后
静态NVPTXCompiler符号外泄，与动态编译插件发生冲突。测试链接恢复相应符号隐藏后，
正常析构与退出均通过；未改agent的清理/所有权逻辑，未跳过析构。
默认OFF agent的动态导出集合与原baseline一致；37个HBF公共头文件及device helper逐字节不变。
[汇总CPU收据](current_machine_20261019/cuda11_compat/CPU_ADAPTER_RESULT_V1.json)，
[注册与launch拒绝收据](current_machine_20261019/cuda11_compat/registry-tests-v3.json)。
正向实际PTX绑定、GPU数值及HBF覆盖仍未验证。

后续冷启动检查实际复现了遗漏：agent早于应用装载CUDA11时，旧NOLOAD收集路径
无法取得11域provider。增量补丁仅在新CUDA11开关内预先取得本地provider，并核对
真实11.0版本符号；11注册转发直接使用该provider。12/13的查找逻辑不变。
修复前冷启动失败、修复后ON/OFF冷启动及完整注册fixture通过，伪装为11库名的
CUDA12 provider被明确拒绝。默认OFF导出仍与原baseline一致，ON未新增导出。
两份新源码的增量补丁重放及最终文件哈希也一致。
[最新CPU汇总](current_machine_20261019/cuda11_compat/CPU_ADAPTER_RESULT_V2.json)，
[冷启动修复收据](current_machine_20261019/cuda11_compat/PROVIDER_FIX_V2.json)。
验证时仅暂停本任务Torch编译158.154秒，之后确认恢复；没有GPU操作。

原Torch wheel无嵌入PTX；其实际git版本`e4ee3be4063b7c430974252fdf7db42273388d86`
和72个递归子模块均已精确核对，tracked diff为空。
前三次及有界重试失败保留；通过同一官方仓库URL后缀/大小写形式获取原提交，
没有改源码、gitlink、`.gitmodules`或全局Git配置。
[源码闭合](current_machine_20261019/cuda11_compat/framework-EXACT_SOURCE_CLOSURE_V1.json)。

CUDA11.7配置通过，239组生成NVCC flags均含sm80 cubin和compute80 PTX，
C++ ABI0与原wheel一致。BLAS为OpenBLAS，原eager算子源码未替换；未启用分布式、
cuDNN、fused attention等可选后端。未消费CMake字段明确记为NOT_USED，
不将它们当作有效开关。
[实际配置审计](current_machine_20261019/cuda11_compat/framework-CONFIGURATION_AUDIT_V1.json)。

框架源码构建已启动：session8496，worker3302057/supervisor3301559，4个逻辑CPU、
nice10，GPU任务数0。内部看护设4小时watchdog、主机可用RAM保留64GiB、
进程组RSS96GiB、任务树75GiB及磁盘可用50GiB停止条件。这是隔离软件构建计划，
不是科学实验或GPU运行授权。完成通知未配置；DONE/FAILED路径及命令见
[当前构建记录](current_machine_20261019/cuda11_compat/framework-build-v1.running.json)，
[资源计划](current_machine_20261019/cuda11_compat/framework-build-v1.plan.json)。
尚未声称wheel构建、实际嵌入PTX检查、GPU数值或HBF Timing完成。

CPU后处理worker3316283/session83284已配置为等待构建完成：成功后仅在新私有prefix
安装本地wheel、核对实际环境和原始输入、运行原helper固定CPU forward、收集实际
libtorch_cuda PTX及编译剩余HBF/agent工件。它不会启动loader、daemon或GPU workload。
完成标记为`/var/tmp/hbfsim-yjl/llama2-layer15-full-v1/logs/postbuild-v1/DONE`
或`FAILED`；未配置外部完成通知。
[后处理资源计划](current_machine_20261019/cuda11_compat/framework-postbuild-v1.plan.json)。

版本副本适配器支持本次实际7.4/7.7来源，只改版本token，保持sm80，原件只读、
输出不覆盖；3项固定CPU规则检查通过。原始transform/compile结果保留，
使用新副本时须建立派生provenance和独立编译验证，不改旧收据。
[副本规则检查](current_machine_20261019/cuda11_compat/VERSION_COPY_CPU_V1.json)。

完整原始库收集：[CUDA11_ORIGINAL_COLLECTIONS.json](current_machine_20261019/cuda11_compat/CUDA11_ORIGINAL_COLLECTIONS.json)。
cuBLAS231个sm80/PTX7.7成员，cuBLASLt88个sm80成员（84个7.7、4个7.4）。
仅证明原PTX来源完整性，不证明本workload实际kernel选择、版本适配后的语义等价或完整覆盖。

## 后续验收准备与限制

只读CPU检查器`experiments/llama2_sm80_cuda11/layer_acceptance.py`已准备，但GPU包装器
尚未接入。它要求九个独立权重scope的地址证据、正常服务路径下的access/byte闭合、
完整请求完成、实际strict bridge patched记录，以及native/HBF FP16输出逐字节一致。
六项合成CPU fixture通过，覆盖缺权重、错误/溢出、错地址、请求未闭合、原生旁路、
固定延迟路径及非有限/不同输出的拒绝；没有实际GPU收据被验证。
[固定检查收据](current_machine_20261019/cuda11_compat/LAYER_ACCEPTANCE_CPU_V1.json)。

源码消费者审计发现，`hbf_device.cu:1235`的EvalDelay Apply分支在正常服务分支之前，
实际替代`resolve_fast_or_hybrid/resolve_leader`；其trace不能证明普通HBF服务完成。
现有access accounting只给module/aggregate计数。计划通过九个单参数模块的独立
forward scope，结合gate实际地址与正常服务计数验证逐权重覆盖；尚未实际接入运行。
这项准备不改变生产helper或服务路径，也不将addressed证据等同于整个权重每字节均读取。

后续仍须完成：实际wheel与嵌入PTX验收、实际kernel/容器发现、明确的staged provenance
和绑定、完整包装及新的GPU预检。cuBLAS本机DSO没有动态cudart依赖，不能假定其内部
Driver路径天然受CUDA11公共host-stub身份监听覆盖；需要实际捕获证据再判定绑定方式。
不以PTX清单数量、CPU成功、旧样本或汇总计数替代这些检查。
