当前更新：用户已回复“批准”，隔离实现与固定CPU验证已完成。见[实际结果](CUDA11_COMPATIBILITY_PROGRESS.md)。以下为提交确认时的原方案；原文与原hash副本另存，GPU尚未运行。

# CUDA11 完整层接入最小方案（待用户确认）

状态：PENDING_USER_APPROVAL；本方案尚未实施，不是GPU实验批准。

原始workload已在`/mnt/hbfsim-workload`找齐且哈希一致，模型
`/mnt/llama2-7b`第15层全部tensor也与归档一致。
原脚本CPU validate-only通过。此前GPU样本仅为Driver API直调cubin，
没有经过Torch运行库、实际gate或agent，不代表完整层兼容。

## 已确认的接口缺口

- CUDA11.7的cuBLAS提供sm80 PTX；当前cu12包提供的对应PTX为sm90。
- 当前gate的`RuntimeDomain`只有Cudart12/13：
  `src/cuda_runtime/launch_gate.cpp:2255`，其运行库选择与版本符号查找也只覆盖12/13。
- 已有agent支持权重补丁同样按12/13维护注册身份和native binding；
  不能将CUDA11调用伪装为12，或靠改.so文件名绕过。
- 本机CUDA11.7确实提供`cudaGetFuncBySymbol`，不需要为此重写kernel发现机制。
- 当前Torch wheel自身没有PTX；需要从其实际git版本重建原框架的sm80/PTX产物，
  不能用自写/CUTLASS算子替换后宣称原路径复现。

## 请求确认的改动范围

1. 仅在独立实验源码副本中增加默认关闭的CUDA11运行库兼容选项。
   现有仓库生产gate/agent源码、默认构建及公共HBF ABI保持不变。
2. gate副本增加`libcudart.so.11.0`域、版本符号解析和对应launch入口，复用现有
   `cudaGetFuncBySymbol`及CUfunction检查，不增加名称猜测、静默fallback或宽松放行。
   只支持本workload实际需要、且CUDA11确实存在的入口；未知入口继续拒绝。
3. agent副本基于原固定ec26源码与既有0002补丁，在相同默认关闭选项下扩展
   CUDA11注册身份、运行库域和native binding。原12/13域逻辑保持原有条件。
   不重写调度、事件、请求/完成生命周期、资源所有权、cache或timing futures。
4. CUDA库原文件不修改；从匹配的CUDA11库提取原PTX，对独立staged副本做受检的
   `.version`适配，始终保持sm80目标。使用真实容器/entry/源PTX/staged身份绑定，
   不将旧样本或其它设备manifest冒充本层实际kernel。

“位于独立目录”不是免确认理由。本方案涉及运行库接口和注册身份适配，
依[AGENTS.md](../../AGENTS.md)的最小侵入规则，在得到明确确认前不实施上述改动。

## 验证与验收

- 固定CPU检查：CUDA11真实版本符号存在、正负域匹配、注册身份与解绑边界、
  native binding拒绝错误身份；12/13既有相关测试回归。无GPU的默认关闭构建检查。
- 当前主机独立agent构建profile须实际验证；原主机build lock不改写、不套用PASS。
- 框架重建保留原Torch git身份和原算子源码，明确记录CUDA11构建及实际库版本。
  不声称与原wheel逐字节相同，实际kernel必须重新收集。
- 完整层GPU执行另做版本/hash绑定预检：native输出对照、全部9个权重
  （404,766,720B）的逐项addressed/active modeled覆盖、请求完成与错误闭合。
  若不能全部覆盖就报告缺口，不称为完整验证通过。
- 任何超出上述接口扩展的生命周期/所有权重构、替换算子、宽松覆盖或额外物理
  假设均停止该分支并另行确认。旧失败、raw、400K与盲测边界不改动。

本方案确认只批准上述隔离兼容实现与固定CPU验证，不包含GPU启动、driver/系统环境
改动、push/merge。输入校验、隔离依赖准备等独立工作可继续。
