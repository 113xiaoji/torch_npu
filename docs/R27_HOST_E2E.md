# R27：实际 E/EI 多流接入与 host-sim 结果

**已经接入实际执行路径；本次原始 DLRM 的 host-sim E2E 没有提升。**
E、EI 的配对延迟分别增加 **43.25% / 43.18%**，均为 **12/12 轮更慢**。
模型 forward、权重、输入及原生算子体保持原样；本次不再以 shadow 计数代替调度。

## 端到端结果

55 / node04，AArch64，原生 Python 3.14.6t（自然 GIL=false），Torch 2.13 CPU。
固定 8 个物理核 `144,146,148,150,152,154,156,158`，NUMA1、单 LLC；
Torch intra/inter-op 各 1 线程，三组均使用 `TASK_QUEUE_ENABLE=0`。
主机共享使用，不宣称独占 CPU。

同一 native 候选和 SDK007，12 轮，每轮轮换 off/E/EI 的顺序；六种排列各两轮。
每组先做 8 次数值验证，再预热 50 次、计时 3,000 次，共 **108,000 次 timed forward**。
另有 3 个独立 cold 进程，未混入稳态统计。计时从原 `model(*inputs)` 开始，
到 `torch.npu.synchronize()` 返回；构造、导入、校验和快照不计入样本。

| 模式 | 跨轮 median 的中位数 | 跨轮 p95 的中位数 | 配对延迟变化 | 更快轮数 |
| --- | ---: | ---: | ---: | ---: |
| off | 1.194113 ms | 1.221390 ms | 基准 | — |
| E：Embedding 双流 | 1.712440 ms | 1.744720 ms | +43.25% | 0/12 |
| EI：再接入 Interaction 消费流 | 1.711245 ms | 1.754840 ms | +43.18% | 0/12 |

百分比是逐轮 `candidate/off - 1` 的中位数，不是表内汇总值相除。
配对 p95 延迟增加 E **42.76%**、EI **43.82%**。
36 个计时窗口为 3.581–5.173 秒；39 个 cold/perf 进程均成功退出，进程组已关闭。
全部数值校验与原 CPU 模型一致，最大绝对误差 **0.0**。
Cold 首次 forward 为 off 2.852290、E 3.675510、EI 4.018130 ms，仅作首次调用描述。

## 确认走了实际多流

生成的 native `Invoke` 现在进入 `ImmediateRuntime`，在调用线程立即执行原算子体，
切换其实际目标 stream，并通过事件维护依赖。没有保存或延后引用捕获的 lambda，
也没有替换模型为 Python dispatcher 实现。

SDK 端独立统计实际执行句柄，按计时窗口做差：

- off：全部算子在 caller stream（本次模拟句柄 32）。
- E：26 次 EmbeddingBag 分到 64/80 两流，各 13 次；Interaction、MLP 留在 caller。
- EI：Embedding 同样 13/13；两次 Cat、一次 BMM、一次 Index 在独立消费流 96，MLP 留在 caller。
- 每个 forward 的原 SDK 数值调用均为 EmbeddingBag26、Cat2、BMM1、Index1、Addmm6、Relu5、Sigmoid1。
- SDK 累计诊断中的 CPU 数值 kernel callback 并发峰值由 off 的1增为 E/EI 的2；
  该峰值包含校验和预热阶段，不是只取计时窗口的峰值。

这些是 CPU host-sim 的实际并发证据，不是 NPU device timeline。

## 没有净收益的开销证据

下表为每次 forward 的跨轮中位统计；CPU 时间包含进程内所有线程，不能当成主线程独占耗时。

| 指标 | off | E | EI |
| --- | ---: | ---: | ---: |
| 进程 CPU 时间 | 1.965 ms | 2.659 ms | 2.754 ms |
| 模拟队列任务 | 42 | 229 | 272 |
| SDK event record | 0 | 183 | 222 |
| SDK event wait | 0 | 4 | 8 |
| 主动上下文切换 | 32.42 | 153.19 | 175.99 |

原数值 kernel 数仍是 42。新增任务主要为事件记录与等待；调度器自身每 forward
记录 E27 / EI31 个事件，其余记录包括框架的跨流分配器生命周期处理。
多流管理增加了 CPU 工作和线程调度，在当前小算子规模下没有获得端到端净收益。
以上没有把总延迟强行拆分成互斥阶段；各项独占耗时尚未分别归因。
EI 与 E 的差异很小，也没有显示稳定的额外收益。

新 SDK 执行真实 CPU 算术，与旧 R25/R26 NoOp 不同，不能拿旧约 0.813 ms 直接当此次基线。
这里的 1.194 ms 才是相同候选、相同数值模拟后端的 off 对照。

## 正确性和生命周期

- E/EI 各 18 个正常场景，共 36 个：变化形状/表数、共享权重、空 bag、padding、
  空整数 Index、别名、原地修改、resize/set、显式同步、foreign caller 等，全部通过。
- 16 个独立故障进程：提交前/后 record、wait 失败，持续 query 失败、旧输出清理等通过。
  无算子重放；完成不可证时保留 7 个 owner，清除测试故障并确认完成后才释放。
- 发现并修复实际 allocator handoff 错误：只登记 producer 时内存提前复用，读到99而非41；
  现在每个输出同时登记 caller0。新候选中，caller 暂停时目标块保持 `active_pending_free`，
  64 次 producer 分配未复用该地址，解冻后数值正确。
- 发现并修复 post-submit 异常析构 abort：guard 在内部 scope 中释放引用，设备/主机
  allocator 的7处 event bookkeeping 使用专用 scope。旧路径真实 RED，新候选 GREEN。
  公共 event 仍拒绝失败域；这些 scope 只屏蔽错误的调度边界重入，保留既有 SDK 错误处理。
- Linux Release/ASan/TSan：实际协议、25个输出故障场景、析构、7处 allocator 调用、
  标量资格及流队列1,000轮并发压力测试通过。队列测试使用 `simulator-source-005`
  的 `runtime.cpp/h` 快照，两文件与 SDK007 的构建输入 hash 完全一致；没有声称
  整个 PTA/数值 SDK 或真实 CANN 已获得全量 sanitizer 认证。
- 本地测试83项：82通过、1项缺本地原模型路径而跳过；原模型已在55实际执行验证。

## 代码与资格范围

生产改动位于 `torch_npu/csrc/core/npu/eager_parallel/ImmediateRuntime.{h,cpp}`、
`ImmediateProtocol.h`、`ImmediateInvoke.h`、`TensorVisitor.h`、`AllocatorEventScope.h`，
以及既有 entry、stream/event、copy、mutation、allocator 边界。
`torch_npu/_eager_parallel.py` 提供显式 `mode="e"/"ei"`，默认保持关闭。
`tools/hostsim_r27` 是独立测试 SDK；`benchmarks/eager_parallel_r27` 是本次测量入口。

首次执行域为单 caller、推理、device0/default caller stream、原生 allocator/base format、
queue0；不接纳训练、graph capture、OpHook/NPUTrace、原生裸 stream/DataPtr 操作、
IPC、allocator checkpoint 或未审计的回调。未知算子先汇合再执行原路径。
现有冻结式 `DependencyScheduler` 保留独立：通用 wrapper 的输出在 body 后才产生，
新协议明确使用预留事件/owner的 begin→立即执行→complete，未冒充预先冻结输出。
新输出的字节数在 body 后才可知，超过保留预算时立即 drain，不继续接纳下一算子；
`byte_limit` 不约束单个原始算子本身的瞬时分配峰值。

**真实 NPU/CANN 数值、设备重叠与性能仍未获资格。** 模拟 Embedding 辅助输出采用
明确的测试策略；未把其值声明成 CANN 的辅助输出合同。模型之外的 CPU 参考仅用于对照。
因此当前开关不应自动启用；本次结果也不支持“多流一定加速”。

## 可复核身份与原始证据

原模型/状态/输入 SHA256 分别为：

```text
model  9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5
state  6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19
inputs 239d00eb84bda9298d9141334f617fb03f46e8ec6d5f764e4ec953b11714d3a6
native 64cbe0cbc95dee6280d8ad57555cf0365ff3a27014bbbad95dd4c9bf99b8226a
_C     a55bd3566d78d68a63471e68b1cacad25148b58476268b9a15f8751e5f75f415
runtime-source 44b3c8bcea87ff7a43c7e3ff473036a2e1710819021193fbcbd5ffaff2aa2a48
final-build-receipt 8834dc7bcd5abeaccde06ce16248ac440c70f93c9494a399e426c985c8767bf9
```

完整 fresh native/_C 构建后，两次局部修补的重编和 source posthash 校验均有独立回执；
初始构建、失败测试与修补前回执保留。最终39个测量进程加载相同二进制和 SDK007。

55 上根目录为 `/home/r27-pta-ei-20261031`：
`e2e-cold-002`、`e2e-perf-001`、`agent-native-normal-004`、`agent-native-faults-002`。
证据包 `validation-final-001.zip` 含256个有hash的成员及manifest，SHA256：
`9ab450d827ade0af01125307afdc7cfb9f0441ab5b5f986bb440caabfebb3c27`。
本地副本位于工作区 `.cache/r27-transfer/validation-final-001.zip`，未把原始日志或二进制提交到仓库。
独立审计重新计算全部108,000个样本、warmup/timed计数、配对统计、文件身份及进程关闭，结果一致。
