# R33：真实 A3 的阶段耗时与 PTA 优化位置

主要退化来自 Host 管理：跨流 allocator 事件被放大，普通 eager 的 PTA 异步队列又被当前 R31 路径关闭。设备侧已经发生并行，但节省的忙时不足以覆盖新增 Host 开销。末尾 synchronize 只占几十微秒。

模型、权重、R31 MetaSyn26 输入、R32 空输入修复和正式原生库都不变；原生 SHA `90196668f98f55945a8910979ac8ae5e0619f6bfd68d257b0f803bacbb43a11e`，输入 SHA `c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1`。本轮没有应用新的生产优化。实际在 A3 物理设备14、同一驻留容器内执行，Python3.14.6t/Torch2.13/NumPy2.4.4、自然GIL=false、CPU8..15和Torch intra/inter-op=1保持原配置。原负载 B128/T26/V1024/D16 的规模限定了这些结论。

## Host 墙钟分解

以下是典型采样窗口每次 forward 的均值，单位 ms。对每组的全部窗口按 E2E 均值排序，取中间窗口；偶数窗口取中间两组整体分项的平均。选择整组分项保留可加性，所有原始窗口仍保留。这组分项不是上一轮 R32 正式 E2E 中位数的替代品。

| 方案 | Host 准备＋提交 | 事件管理 | 内部同步/拷贝/轮询 | 末尾 synchronize | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 2.0243 | 0.0000 | 0.0497 | 0.0503 | 2.1244 |
| E | 2.8853 | 0.8034 | 0.0879 | 0.0680 | 3.8446 |
| EI | 2.9918 | 0.9811 | 0.1015 | 0.0670 | 4.1414 |
| 选择性 E | 2.4129 | 0.1045 | 0.0787 | 0.0612 | 2.6574 |
| 选择性 EI | 2.4496 | 0.2680 | 0.0642 | 0.0622 | 2.8439 |

| 方案 | Host 准备＋提交占比 | 事件管理占比 | 同步合计占比 |
| --- | ---: | ---: | ---: |
| off | 95.3% | 0.0% | 4.7% |
| E | 75.0% | 20.9% | 4.1% |
| EI | 72.2% | 23.7% | 4.1% |
| 选择性 E | 90.8% | 3.9% | 5.3% |
| 选择性 EI | 86.1% | 9.4% | 4.4% |

Host 准备＋提交包括 Python/ATen/PTA 包装、参数/缓存 key、allocator 元数据、资格检查、Storage 保留、SDK 提交及未单独截获的 Host API。它是扣除已测事件/同步后的剩余 forward 时间，不能全归因为某一个审计函数。嵌套 API 使用 exclusive 时间，避免重复相加。最终同步使用完整 Python 调用尾部墙钟，内部同步包含同步 CPU 索引拷贝。

直接测到的暖态 SDK `GetWorkspaceSize`、`aclCreateTensor` 均为0次/forward；冷态正对照能捕获这些调用，例如EI首次forward有24次Embedding Workspace和236次CreateTensor，证明截获链路有效。固定输入已经命中executor cache。SDK执行/提交约0.56–0.59ms；其余Host约1.45–2.39ms，分散在Python、PTA、缓存及allocator等路径，不能把整个数值叫作“CANN准备”。

## 事件放大的具体来源

| 方案 | 实际 RecordEvent/forward | R31调度器 | allocator/EventTask | QueryEvent/forward |
| --- | ---: | ---: | ---: | ---: |
| off | 0 | 0 | 0 | 0.0 |
| E | 183 | 27 | 156 | 156.0 |
| EI | 222 | 31 | 191 | 191.2 |
| 选择性 E | 16 | 4 | 12 | 12.4 |
| 选择性 EI | 56 | 9 | 47 | 47.3 |

实际计数来自45个typed CANN/同步API的进程内转发计时，设备时间线的EVENT_RECORD数独立确认一致。调用PC用正式库的符号信息还原：调度器是`ImmediateProtocol::begin/commit`，额外记录是`EventTask::LaunchRecordTask`，查询是`NPUEvent::query`。源码中`NPUCachingAllocator::insert_events`按每个block的每个登记流分别record，并在`process_events`逐个query。

`DeviceCachingAllocator::recordStream`目前无条件把流加入`stream_uses`，包括该block的分配流本身；同文件`BlockComparatorSize`按分配流区分内存池。R31又为输出登记producer和caller，EI再登记consumer。当前SUM有三个非空输出/表，offset2bag为空，形成可核对的计数：

- E：`26 × 3 × 2 = 156`个allocator记录。
- EI：上述156，加第一次Cat的27个输入登记、4个Interaction输出各两次登记，合计`156+27+8=191`。
- 选择性E：`2 × 3 × 2 = 12`。
- 选择性EI：`12+27+8=47`；24张小表仍在caller，其结果跨到consumer也需要登记。

因此只看R31自己的27/31/4/9个事件，遗漏了大部分成本。EI典型窗口事件API约0.98ms，其中RecordEvent约`0.872ms`，查询约`0.083ms`。这些数值还不包含所有allocator容器/锁/上下文处理，属于该路径成本的直接可见部分。

最初重点检查了50µs轮询：实际平均每forward仅约0.09–0.25次sleep，对应9.8–26.6µs；单次实际休眠约106–108µs。它会增加延迟，但在当前负载中远小于事件放大和其余Host成本。

## NPU 计算与重叠

设备时间来自独立NPU-only Level0采样，每组100帧、每帧严格验证145个计算kernel，其中100个是EmbeddingBag内部Cast。下表的忙时与Host工作重叠，不能再加进上面的Host墙钟分解。

| 方案 | kernel时间和 ms | kernel忙时并集 ms | kernel重叠量 ms | 设备采样帧内计算活跃比例 |
| --- | ---: | ---: | ---: | ---: |
| off | 1.2758 | 1.2758 | 0.0000 | 54.2% |
| E | 1.3491 | 1.2341 | 0.1150 | 28.0% |
| EI | 1.3683 | 1.2568 | 0.1115 | 26.3% |
| 选择性 E | 1.3000 | 1.1975 | 0.1025 | 40.2% |
| 选择性 EI | 1.3111 | 1.2017 | 0.1093 | 38.0% |

比例为每帧`计算kernel区间并集/采样帧墙钟`再取平均，表示时间上的活跃，不是芯片峰值算力利用率。多流重叠约0.10–0.12ms；考虑各kernel实际时长，忙时并集比off仅少约0.02–0.08ms。这个量级覆盖不了Host额外的约0.5–2ms。

25个非空EmbeddingBag每个产生1个Embedding计算和4个Cast，另有1个空SUM补零、12个MLP及7个Interaction kernel。详细pilot确认两个输入INT64→INT32、两个辅助结果INT32→INT64的转换，共100个Cast/forward，约0.098ms设备执行时间；它们还增加了提交次数。这是算子相关的成本，后续适合检查等价的推理接口/融合实现，并保持原输入、返回dtype和异常语义。

Host与device的epoch偏差不能假定为零，E样本中偏差超过156µs，直接按Host区间截取会漏掉kernel。本工具使用原模型同步帧末尾Sigmoid的设备顺序分组，每帧145个kernel完整，再以“计算不能先于forward、必须在同步返回前结束”建立常量偏移的因果上下界。按这些界计算，末尾同步中的计算kernel忙时平均上界仅约2.4–4.7µs。绝大多数设备计算已与forward内Host工作重叠。

## 原生任务队列对照：PTA 已有的机会

R31当前要求`TASK_QUEUE_ENABLE=0`。额外的off-only对照只将该参数改为1，保留同一包、模型和输入，三次独立进程配对均通过全部数值/关闭门禁：

| 配对 | off队列关闭 ms | off队列开启 ms | 配对耗时变化 |
| --- | ---: | ---: | ---: |
| 1 | 2.0921 | 1.7176 | -17.90% |
| 2 | 2.2241 | 1.7060 | -23.30% |
| 3 | 2.2427 | 1.7394 | -22.44% |

配对中位耗时降低`22.44%`，3/3次更快；各窗口均值再取中位数分别约2.224ms和1.718ms。原生队列能够把部分CANN提交与调用端工作重叠。这个off对照不等于E/EI已经支持队列1：当前同步Host提交协议中，kernel与完成事件的提交顺序依赖队列关闭，兼容工作必须让二者在同一消费序列中正确发布。

## PTA 层的优化顺序

1. **先验证同分配流登记的短路。** 在当前已准入的TQ0、非捕获路径，依据allocator中真实`block->stream`判断同流，保留所有跨流登记。这比单纯增加worker更直接，位置是`NPUCachingAllocator.cpp::recordStream`及`ImmediateRuntime.cpp::record_use`。内存池同流复用的FIFO条件、allocator种类和错误路径都须作为门禁。当前E计数中的78个producer同流登记是明确的验证目标，速度收益尚未实测。
2. **将逐block事件改为有界批次完成fence。** 在`clear_owners/insert_events/process_events`，按流聚合已退休blocks，批次末尾发布一次fence，查询一次后退休整批；block在所有相关流完成前保持不可复用。事件句柄、失败隔离、alias/view、并发释放、OOM与capture边界都要保留。该方案与模型结构解耦，目标是allocator层重复记录/查询。
3. **兼容原生异步任务队列。** 调整`AclBackend`与`OpCommand::RunOpApiV2/AsyncTaskQueueInterface`的衔接，把kernel、依赖wait和completion record作为有序任务发布，并由设备完成fence驱动异步回收。保留设备完成证明；以此替代caller上的整流完成等待。下轮同时保留off(TQ0)因果基线和off(TQ1)实际框架基线。
4. **再压缩重复Host包装。** 给正常caller算子提供仍会正确关闭待执行组的轻量入口；一次入口内复用已审计的Storage/元数据，减少重复句柄拷贝与状态读取。CPU采样显示热点分散，`retain/recordStream/native_tensor`均出现，但并无证据可把全部剩余Host时间都算作资格审计。暖态缓存已命中，GetWorkspace/descriptor的并行准备不是本负载的优先目标。

本轮提交计时工具和分析；上述生产优化尚需实现与验证。验证应覆盖原五组数值/路由、allocator提前复用、跨流alias、事件重录代际、外部引用、并发释放和错误回收，再跑多类型模型和A3 E2E。当前小规模MetaSyn投影的结论不替代更大embedding表/动态shape的测量。

## 采样、复核与范围

主采样16个进程、EI及任务队列复测7个、独立CPU采样2个，共25个模型进程、81500个计时forward，均保持原模型/input/native pins及GIL=false。完整窗口包含before/after native计数、数值、关闭和原始逐帧时长。`calibrate.measurement`只是最后窗口的重复引用，样本总数只计`windows`一次。

API计时按调用线程TLS记录嵌套exclusive时间，使用100MHz CNTVCT；从真实SDK头生成typed转发函数，并验证该任务没有需要特殊转发语义的RTLD_NEXT查询。开启/关闭计时配对的中位变化为off+0.06%、E+0.31%、EI+1.20%、选择性E−0.44%、选择性EI+0.17%。EI有一段+38.59%的波动，全部保留，并新增三对窗口；未删除原始异常窗口。无preload对照也单独保留，跨进程差异不全归因为计时器。

完整CPU+NPU profiler的pilot把off均值从约2.24ms推到3.28ms，因此没有拿它做正常Host占比分解。仅设备采样仍增加约8–14%的墙钟，设备表为该独立采样cohort。采样级别与CANN层/设备层数据说明见[官方CANN9.1 Profiler接口](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html)。

另外在两个未加载API计时器的4000次工作进程中采集主线程用户态CPU样本（499Hz，零丢样）。self DSO占比off为Python30.26%、PTA17.91%、libc12.69%；EI为Python19.67%、PTA18.10%、libc23.74%。这些是用户态CPU采样比例，不能直接当成墙钟占比。原始perf.data保留；最初inline符号解析耗时过长，停止该离线解析后用no-inline/self-only恢复，模型进程数值与生命周期检查已完成。

区间算法单测覆盖ns级epoch精度、重叠并集、重复分析lane、Level0类型及设备顺序分组。事件API计数又由独立硬件记录核对。[结构化结果](R33_REAL_A3_PHASE_RESULTS.json)保留全部Host窗口、扰动、设备摘要及任务队列配对。

证据包`r33-a3-phases-evidence-001.zip`：72321493bytes、561个SHA/size标记payload，SHA`7567601e598518a96ee4cea9d7f3d6756b1adcf9a180d656b89a8f0a3a79d46e`；包含完整原始样本、trace/原始profiling、API tap源码/二进制、SDK声明manifest、perf数据和执行回执。生产原生库未放入证据包。现场在A3的`/home/zyz/r33-a3-phase-20261111`，本地副本在项目根`.cache/r33-transfer/`。
