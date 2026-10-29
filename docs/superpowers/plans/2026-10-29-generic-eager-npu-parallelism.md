# Generic Eager NPU Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留模型 forward/API，在经过资格验证的算子域内提高真实 NPU 性能，并把线程同步正确性、可观察语义与收益列为独立硬门槛。

**Architecture:** 首版只改 PTA/op-plugin，复用原 CANN kernel，将 Embedding 和 Interaction 联合作为依赖流水线。通用 effect/lease/依赖层采用单一状态所有者，先验证串行 shadow，再启用 Embedding side streams 与 Interaction fan-in；host Prepare、kernel 改造、多表融合均为后续独立选项。

**Tech Stack:** 当前固定 revision 的 PyTorch/torch_npu/op-plugin、C++17、Python std/FT、CANN/Ascend C、CPU oracle、host-sim、真实 NPU。

设计依据：[R26 设计与验收规格](../specs/2026-10-29-generic-eager-npu-parallelism-design.md)。
当前优先级依据：[PTA Embedding＋Interaction 联合设计](../specs/2026-10-29-pta-embedding-interaction-pipeline.md)。
代码基线：`60246fe9a9fff2fc724bf9155f0f38fb4266b0bc`。
本文件是待执行计划；下述新增文件、API、测试命令均须由对应任务实现，不能声称现已可运行或已通过。
仓库路径均相对本 checkout；外部 `<ops_nn_root>` 必须由任务 0 的真实 qualification manifest 解析，不是假定本地存在的路径。

## 执行顺序与独立交付

```text
0 目标资格/证据工具 -> 1 数值合同 -> 3 effect入口 -> 4 lease/mutation
                                -> 5 协议核/模拟 -> 6 PTA联合流水线 -> 真机EI验收
仅在真实kernel瓶颈成立时：任务2 单kernel(K) -> 单独K/O验收
7 性能选择/模型矩阵贯穿各模式；P/F 只有对应前提通过后才有实现任务。
```

首版执行 0、1、3、4、5、6、7，任务 2 延后为可选路线。没有真机时只完成
工具、CPU 与模拟合同，不把 kernel/S 标记为合格。每项以聚焦回归、独立 review、
对应证据和单独 commit 为完成条件；不一次性落地全部模块后再测试。

## 任务 0：固定目标与可归因测量

**新增文件：**

- `benchmarks/eager_parallel_r26/qualification.py`：绑定 SoC、driver、CANN、Python/Torch/op-plugin、源码与加载产物。
- `benchmarks/eager_parallel_r26/contracts.py`：模式、身份、计数、生命周期和统计门槛。
- `benchmarks/eager_parallel_r26/run.py`、`worker.py`：独立进程控制、baseline 与 device/CPU 双计时。
- `test/eager_parallel/test_measurement_contracts.py`：拒绝错产物、漏轮次、错误计时及混用模拟证据。

- [ ] 先写负例：provider hash 不符、未知 SoC、无真实 kernel、少一臂、重复 forward、只测 default stream、把 cold trace 放进 perf 必须失败。
- [ ] 实现 qualification schema；`source_root/revision`、`loaded_libraries`、`sdk_contracts`、`device_kind` 均可追溯。K 资格需要真机与目标内核源码；S/EI 资格需要固定的已加载 vendor 产物与真实 SDK 合同，不要求修改其内核源码。两者都不从 host-sim 填充。
- [ ] 实现 O/K/P/S/F 五种单项模式；同一次实验各臂加载同一候选，开关选择原或新路径。独立记录实际 kernel/API 路由，防止“全部 fallback 却报告加速”。
- [ ] CPU wall 包含原 forward 与完整 join；device event 结束点在所有 stream 完成之后。context-switch/CPU/内存统计位于样本窗之外或以各臂相同的窗口计数读取。
- [ ] 原始 DLRM 固定 source/state/input hash；新增模型清单固定类、版本、shape、seed、state hash。输出 artifact 放新目录，不覆盖 R25。

验证命令（本任务实现之后）：

```bash
python -m unittest discover -s test/eager_parallel -p test_measurement_contracts.py -v
python benchmarks/eager_parallel_r26/qualification.py --config target.json --output qualification.json
python benchmarks/eager_parallel_r26/run.py --config experiment.json --plan-only --output fresh-plan
```

通过条件：所有负例拒绝。K 需要目标 kernel 源码/工具链；PTA-only 的 S/EI 可以复用已资格 vendor kernel，但必须证明真实 SDK/设备合同。O/CPU/host-sim 与两者分别报告。

## 任务 1：EmbeddingBag 数值与接口合同

**新增：** `test/eager_parallel/test_embedding_reference.py`、`test/cpp/eager_parallel/embedding_contract_test.cpp`、
`test/npu/test_eager_parallel_embedding_bag.py`。**参考现有：**
`third_party/op-plugin/test/test_base_ops/test_embedding_bag.py` 与 `EmbeddingBagNpuOpApi.cpp`。

- [ ] 写小型显式 FP64 sum oracle，另比固定 CPU `aten`；主输出与辅助输出定义域分开。
- [ ] 写原始四输出合同，记录同 SoC/SDK 的 eager-off 形状和有效内容，拒绝读取未定义辅助值。
- [ ] 先使下列退化实现失败：丢最后一个 bag、错误处理空 bag、将 defined-empty sample weights 当 None、少创建辅助输出、错误 padding。
- [ ] C++ 覆盖 nullopt、engaged undefined、defined-empty、defined-nonempty；domain 外保持原结果/错误和 fallback reason。
- [ ] 矩阵：B={1,16,128,1024}，D={1,16,31,32,64,128,257}，bag length={0,1,2,8,64,1024}；
  均匀/热点/重复索引，小表/超 cache 表，非连续 view、末维尾块，正负混合/NaN/Inf。
  用预先固定分层抽样而非所有组合；DLRM B128/D16/L2 必须包含。
- [ ] 先资格 FP32/sum；mean/max、其他 dtype、sample weights、训练用作 fallback 正确性负例。

最小 oracle 断言示例：

```python
# W=[[1,2],[3,1],[2,4]], indices=[0,1,2], offsets=[0,2,3]
# include_last_offset=False：bags=[0,1],[2],[]
expected = [[4.0, 3.0], [2.0, 4.0], [0.0, 0.0]]
# 比较主结果与 expected；辅助输出只比较合同有定义的部分。
```

验证命令：

```bash
python -m unittest discover -s test/eager_parallel -p test_embedding_reference.py -v
python -m pytest test/npu/test_eager_parallel_embedding_bag.py -q
```

通过条件：oracle、原 NPU、候选三方一致；确定性开关与 baseline 的行为一致；数值不合格则不测对应优化域性能。

## 任务 2：单 kernel 内并行（可选后续 K）

**修改：** 固定目标的 `<ops_nn_root>/index/embedding_bag/op_host/`、`op_kernel/`、`tests/`；
`third_party/op-plugin/op_plugin/ops/opapi/EmbeddingBagNpuOpApi.cpp` 的 route。
**新增：** `torch_npu/csrc/core/npu/eager_parallel/Policy.{h,cpp}`；
`patches/eager_parallel_r26/` 与固定 manifest（采用 overlay 时）。

- [ ] 保存实际 stock kernel 的 tiling/launch/设备耗时与访存证据；目标源码 manifest 列出实际架构 leaf 文件后才改 kernel。
- [ ] 在外部 kernel 仓先写 tile coverage 测试：每个有效 `(bag,dimension)` 恰好一个最终写者，尾部/空片无漏写或越界。
- [ ] 实现短 bag 的独占输出 tile、连续维读、向量归约；固定归约次序，不新增跨 block 自旋。
- [ ] 长 bag 分段作为独立提交；测试 partial workspace 上界、第二阶段 join 与所有输出；没有收益不启用该策略。
- [ ] 保留原 ACLNN schema；若 custom op 名称不同，只在桥接层按资格表选择，模型仍用原 ATen 调用。
- [ ] 接入 O/K 选择并在诊断中记录实际 kernel 身份；CPU worker=0，原 stream、workspace/事件生命周期沿用已验证路径。
- [ ] 运行任务 1 的真实矩阵、目标设备 kernel race/边界检测（以工具支持为准），再做 K/O 配对性能。

验证：先用固定外部 revision 自带的官方构建/test 入口，实际完整命令写入 qualification 的 `kernel_build_commands`；
随后执行任务 1 的 NPU 测试，以及：

```bash
python benchmarks/eager_parallel_r26/run.py --config kernel-ab.json --modes O,K --execute --output fresh-kernel-ab
```

通过条件：所有资格域数值正确，实际 kernel route 命中，达到设计的 median/p95 门槛。
真实带宽已饱和或 stock kernel 已优时，保留 stock route，不能为满足目标虚报收益。

## 任务 3：通用 effect 入口与串行 shadow 模式

**新增：** `eager_parallel/OperatorContract.{h,cpp}`、`EntryGuard.{h,cpp}`；
`test/eager_parallel/test_entry_coverage.py`、`test/cpp/eager_parallel/effect_test.cpp`。
**修改：** `torchnpugen/{utils.py,gen_backend_stubs.py,custom_functions.py}`、
`third_party/op-plugin/torchnpugen/gen.py`、`torch_npu/csrc/aten/VariableFallbackKernel.cpp`。

- [ ] 写生成代码断言：guard 在 device/unsafe/FormatHelper/`op.meta` 等 Tensor 元数据读取之前。
- [ ] 为 manual、foreach、custom、direct C++ 与内部重入生成入口覆盖清单；未覆盖入口令 S 启用失败。
- [ ] 实现 op schema/effect registry，默认 Unknown；每项显式描述输入读写、输出/alias、metadata mutation、host observation 和 Prepare 是否读数据。
- [ ] 先只登记依赖并沿原路径顺序执行。验证每个原算子只执行一次，内部重入不重复登记外层逻辑效果。
- [ ] 测试 `.item()`、CPU fallback、未知 custom、RNG、autograd 和显式 stream 边界均在读取/副作用前完成必要 drain。

验证命令：

```bash
python -m unittest discover -s test/eager_parallel -p test_entry_coverage.py -v
ctest --test-dir build/eager_parallel -R effect -V
```

通过条件：serial shadow 的数值、异常与副作用次数等于 O；此阶段不宣称 device 并行。

## 任务 4：一次验证、冻结记录和地址租约

**新增：** `eager_parallel/FrozenCall.{h,cpp}`、`AllocationLease.{h,cpp}`；
`test/cpp/eager_parallel/lease_test.cpp`、`test/npu/test_eager_parallel_mutation.py`。
**修改：** `NPUCachingAllocator.{h,cpp}`、`ResizeNpu.{h,cpp}`、`SetNpu.cpp`，
并审核 `LocalScalarDenseNpu.cpp`、copy/storage mutation 的其他入口。

- [ ] 首先写能检出“Storage 强引用仍在，但 DataPtr 已被 resize 替换”的失败用例。
- [ ] 定义 allocation ID/generation、租约计数和边界；验证、捕获 metadata 与取得租约处于同一受保护事务，不能先检查再无保护取地址。
- [ ] 将 eligibility 与 snapshot 共用一次检查产物；深度资格缓存绑定 allocator/config epoch，动态 instrumentation/stream use/capture 变化立即失效。
- [ ] mutation 入口先处理当前 generation 的在途访问；旧数据完成前不得复用；地址复用后旧 ACK 不得影响新 allocation。
- [ ] 实现 unknown alias/外部分配的全域 fallback，必要时关闭 S。不能用 Tensor version counter 代替 allocation 生命周期。
- [ ] 压力验证 view/in-place/out、`resize_`、`storage.resize_`、`set_`、空 storage、地址 ABA 和内存不足。

验证命令：

```bash
ctest --test-dir build/eager_parallel -R lease -V
python -m pytest test/npu/test_eager_parallel_mutation.py -q
```

通过条件：模拟故意提前释放必须被测试识别；真实 allocator 压力下结果及错误一致，未发生提前复用。

## 任务 5：状态机、通知与退休协议（同步硬门槛）

**新增：** `eager_parallel/DependencyScheduler.{h,cpp}`、`CompletionTracker.{h,cpp}`；
`test/cpp/eager_parallel/{state_machine_test.cpp,notification_test.cpp,retirement_test.cpp,CMakeLists.txt}`；
`test/eager_parallel/test_hostsim_scheduler.py` 与可控 fake device adapter。

- [ ] 写最小状态机测试，先用错误实现复现重复 launch、完成前 release、丢唤醒与旧 generation ACK。
- [ ] 实现单 owner 状态变更；S/P 使用专用 coordinator，K/O 无新增线程。每个 worker 独立 inbox/completion。
- [ ] 定义 submit registration ACK、launch receipt、completion ACK、retired ACK；caller/worker 不能直接修改 owner 状态。
- [ ] 使用持久条件/计数通知，覆盖通知先发生、检查空队列到入睡、两个 ACK 交错；不复用 R25 的全状态 `notify_all`。
- [ ] coordinator 在额度不足时仍消费 completion/control，只有 producer 被背压；SDK 阻塞等待不在 allocator/scheduler 锁内。
- [ ] 用 2 workers、capacity=1/2、3 jobs 枚举关键事件交错；对随机 1/2/4/8 workers 跑固定种子压力，检测进展与重复次数。
- [ ] fault injection 覆盖 Prepare/launch/event record/query/wait/destroy、取消/关闭、晚 ACK。已提交失败保留首错和资源；无法确认完成进入 quarantine。
- [ ] event/job pool 复用必须在全部 CPU 借用与设备完成后，generation 单调；关闭后拒绝新 admission。

状态机用例表（测试要逐条编码）：

| 输入交错 | 必须观察到 |
| --- | --- |
| 完成通知先于 waiter 入睡 | waiter 读取持久完成状态，不永久睡眠 |
| cancel 先于 launch | launch 次数 0，CPU 借用 ACK 后释放 |
| launch 先于 cancel | launch 次数 1，直到 device complete 才释放 |
| pool slot 重用后到达旧 ACK | 新 generation 不变，旧消息拒绝并计数 |
| admission 满且所有 work 已完成待退休 | owner 继续处理退休，producer 最终前进 |
| close 与最后 submit 竞态 | 每个 submit 恰好 accepted 或 rejected，无孤儿任务 |
| event 完成未知且取消 | quarantine，不回收，不重跑原算子 |

验证命令（拟新增独立 core CMake 工程）：

```bash
cmake -S test/cpp/eager_parallel -B build/eager_parallel -DEAGER_PARALLEL_SANITIZER=thread
cmake --build build/eager_parallel
ctest --test-dir build/eager_parallel --output-on-failure
python -m unittest discover -s test/eager_parallel -p test_hostsim_scheduler.py -v
```

ASan 使用另一个 build 目录重跑；不能把未 instrument 的 vendor SDK 算作 sanitizer 已验证。
正常用例资源全清零；预期失败用例要求受控隔离、首错稳定、无新 admission。
任何 race/deadlock/lost-wakeup/提前释放均阻断任务 6，不以“压测暂未复现”替代失败修复。

## 任务 6：Embedding＋Interaction 的 PTA 联合调度

**新增：** `eager_parallel/ExecutionAdapter.{h,cpp}`、`torch_npu/_eager_parallel.py`；
`test/npu/test_eager_parallel_runtime.py`。
**修改：** `NPUStream.{h,cpp}`、`NPUEvent.{h,cpp}`、allocator 多 stream 退休、
`StreamAndEventKernelNpu.cpp`、`Module.cpp` 与默认 off 的 build/config 入口。

- [ ] 先写独立双分支与 RAW/WAR/WAW/diamond 的假设备测试；同流顺序继续核 FIFO，跨流核依赖边与 exactly-once。
- [ ] 完成 CANN 契约探针：query/execute thread、目标 stream、executor/descriptor/workspace 生命周期及错误后提交状态。未资格的并发方式保持关闭。
- [ ] 实现原 stream 前驱接入、已 ready 节点 launch、事件 record 提交 ACK、consumer wait；不能把 host record 返回当成已提交。
- [ ] 实现 scope exit、未知 op、D2H/.item/mutation/显式同步的 frontier join；所有输入输出和 workspace 记录 stream 使用与 lease。
- [ ] 先启用两条 stream；原 kernel 不变，host Prepare worker 仍 off。注册 EmbeddingBag 加最小纯算子集，未知效果正确 fallback。
- [ ] 两条内部 side stream 承载独立 Embedding；x 和完整 Interaction 留在原消费 stream。Cat 对实际 producer frontier 做 device fan-in，不能无差别等待 S0 当前 tail。
- [ ] 覆盖 Cat、Bmm、Index、view/transpose/slice alias 以及 MLP 实际路由；补 `AdvancedIndex.cpp` 的 CPU index H2D 与 bool/Nonzero 数据依赖 shape 边界。
- [ ] 新增 `test/cpp/eager_parallel/fan_in_test.cpp`、`test/eager_parallel/test_feature_interaction_reference.py`、`test/npu/test_eager_parallel_feature_pipeline.py`，按补充设计核乱序完成但原表/三角索引顺序不变。
- [ ] 原四输出、x 两次消费、T/transpose只读别名、CPU索引源保活全部纳入 lease；只在明确效果和目标路由资格通过后免除全域 host fence。
- [ ] 真实设备 timeline 证明独立 kernel 区间重叠，依赖节点没有提前读取；再跑 S/O 性能和完整模型矩阵。

验证命令：

```bash
python -m pytest test/npu/test_eager_parallel_runtime.py -q
python -m pytest test/npu/test_eager_parallel_feature_pipeline.py -q
python benchmarks/eager_parallel_r26/run.py --config schedule-ab.json --modes O,S --execute --output fresh-schedule-ab
```

通过条件：真实数值/别名/观察边界、全部生命周期与故障测试通过；依赖宽度足够的已选域有 device overlap 且端到端收益达门槛。其他域正确回退。

## 任务 7：通用性、成本选择与发布

**新增：** `benchmarks/eager_parallel_r26/models.json`、`test/eager_parallel/test_policy.py`、
`docs/R26_EAGER_PARALLEL_RESULTS.md`；扩展 `Policy` 与资格清单。

- [ ] 固定原 DLRM、多分支 MLP、CNN/ResNet、Transformer 的版本/输入/权重；验证不改 forward。
- [ ] 加严格依赖链、饱和大 GEMM/带宽、频繁 `.item()`、alias/in-place、未知 custom 等负例。
- [ ] 冷诊断测 stage：caller capture、admission wait、Prepare CPU/wall、launch、event wait/join、retirement；另测 mutex 等待/通知次数。带 trace 的结果只用于归因。
- [ ] 性能按 O/K/P/S/F 独立对照，组合后置；在线查询策略表必须足够便宜且不重复执行算子来试错。
- [ ] PTA首版额外比较 O、B（新运行时串行）、E（仅Embedding分流）、EI（整链调度）；同一原模型统计 Embedding、EI、完整forward，局部收益不能代替整链收益。
- [ ] P 的资格额外要求共享锁/唤醒压力对照、W1 相对 O 的总成本门槛；SDK 内部串行或 Prepare 很轻的域保持 P=off。
- [ ] F 仅在已经 ready 的同族独立节点上分组，记录 group size/等待预算/flush；保留全部输出和原消费者映射。不为了凑够“26表”等待未来模型调用。
- [ ] 10个独立配对轮次、每进程≥200样本且窗≥2秒；以进程为单位统计。收益≥5%、p95/control回退≤3%、置信范围与噪声门槛按规格执行。
- [ ] 发布资格化模式、fallback原因、源/产物/SDK pins、数值/同步/设备/模型四份结果；不把某个模型加速推广为任意模型。

发布矩阵必须明确：

| 条件 | 发布状态 |
| --- | --- |
| 只有 CPU/host-sim 通过 | 实验功能；真实 K/S 未资格 |
| K 数值与性能通过，S/P 未通过 | 只发布已资格 K 域 |
| S 正确但无稳定收益 | 保持 off 或只允许诊断 opt-in |
| 同步/生命周期任一硬门槛失败 | 禁止该模式发布 |
| 目标域收益通过、负例正确回退 | 才允许进入显式资格策略表 |

## 交付审计与当前状态

- 设计阶段已完成本仓库改码位置核查、独立代码地图与独立测试方案调查。
- 后续 PTA-only 联合修订已对设计、联合流水线和计划三份文档完成独立只读审阅，
  未发现阻断项；补齐 vendor 资格/内核源码的区别与分阶段数值门槛。
  该结论仅针对文档设计，每个实际协议提交仍须独立代码 review 和对应测试。
- 当前没有 R26 生产改动或新性能承诺。本文列出的测试是实施验收项目，尚未执行。
- 实施工作首先做任务 0/1，并在 CPU 推进 3–5；有真实 NPU 且 vendor API 合同时
  推进 6/7，不要求先改 CANN。任务 2 由后续 kernel 瓶颈证据触发。
  不能在 host-sim 上宣称完成真实查表/求和/Interaction 的优化。
