# A3 Eager Prepare 性能对照与归因

**D894 候选在本次固定混合输入、热缓存、单 Python caller 场景中没有提速。**
MPSC4 首轮中位延迟4.527ms，off为1.701ms；关键组反向复测分别为4.408ms和1.712ms。
MPSC4 比同样4worker的加锁入口慢32–37%。这些结果对应已测候选，不覆盖其他工作树中的未提交优化。

运行时是Python3.14.6 free-threaded build，导入Torch/torch_npu后GIL仍关闭。
Torch2.13.0+cpu git `cf30153…`、NumPy2.4.4、真实CANN9.1、A3物理14映射逻辑0，CPU8–15，
Torch intra/inter-op均1，使用FP32且关闭HF32。主对照为一个Python caller和1/2/4个原生C++ Prepare worker；
四个Python caller仅在之前的保序正确性资格测试中验证，未在这里做并发forward吞吐比较。

模型、权重和输入四项SHA保持不变：原版DLRM，B128、26表、V1024、D16、50850个ID，
25个非空表及原始空表，sparse=True。所有组用同一候选库，TQ1、同一默认流和原PTA FIFO。
每个性能窗口50次预热、3000次原始forward加设备同步；数值检查、快照、关闭和启动均不计入单次延迟。

## 未插桩结果

首轮完整七组，单位ms/forward。CPU时间累计了进程内各线程，可以大于实际延迟。

| 方案 | 中位延迟 | p95 | p99 | 累计CPU时间 | 自愿切换/forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| off(TQ1) | 1.701 | 1.721 | 1.731 | 2.785 | 95.47 |
| mutex1 | 2.595 | 2.726 | 2.786 | 6.850 | 336.04 |
| mutex2 | 2.865 | 3.007 | 3.067 | 8.015 | 567.61 |
| mutex4 | 3.295 | 3.453 | 3.551 | 10.598 | 1095.70 |
| mpsc1 | 3.180 | 3.383 | 3.515 | 8.824 | 714.35 |
| mpsc2 | 3.503 | 3.745 | 4.088 | 10.217 | 1059.87 |
| mpsc4 | 4.527 | 4.930 | 5.672 | 14.460 | 1741.98 |

随后按mpsc4→mutex4→off反向顺序复测关键三组，仍各50预热、3000计时：

| 方案 | 中位延迟 | p95 | p99 | 累计CPU时间 | 自愿切换/forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 1.712 | 1.751 | 1.840 | 2.812 | 95.16 |
| mutex4 | 3.340 | 3.496 | 3.608 | 10.759 | 1121.95 |
| mpsc4 | 4.408 | 5.146 | 5.749 | 14.156 | 1724.18 |

反向复测相对首轮中位数变化为off+0.64%、mutex4+1.37%、mpsc4−2.62%，退化方向保持。
这是一个完整七组pilot和三个关键窗口复测，共10个窗口、30000次计时forward，
并非两轮完整七组或14轮平衡实验，不能外推所有模型、动态shape或多Python caller的性能。

每个enabled窗口确实执行75000次真实SDK Prepare，记录54000次fallback fence；
关闭时76475个ticket全部执行并退役，零retained/pending、无错误，数值、顺序及关闭ACK均通过。
不是零准入、shadow或缩小输入得到的结果。

## 为什么没有收益

**热缓存后，可分摊的重查询工作消失了。** 对相同D894二进制做独立外部uprobe诊断，
50次预热后的100个forward中，两种入口的2500次worker cache lookup全部命中，
EmbeddingBag GetWorkspaceSize调用均为0。探针在各自冷阶段均捕获89次query，确认不是漏探针。
四个Prepare worker均有工作，但这段热态Prepare区间重叠峰值只有1；此前冷两forward诊断曾达到4。
因此本负载的热态任务很短，不能把冷态重叠直接当作稳定模型加速。

**单caller没有原提交锁的生产者争用，MPSC增加了发布往返。** 原加锁入口的submit_mu在这里没有
多个caller同时竞争。MPSC仍等待publisher函数完整返回，当前caller才能提交下一请求，
ingress不能形成多请求批量优势。每forward25个请求仍各有一次交接及确认。
原off的TQ1本身已有caller准备下一op与consumer/device执行上一op的异步重叠。

**快照和生命周期管理并未被worker分摊。** 每个合格Embedding在domain和snapshot阶段分别检查7个Tensor，
25个任务约350次资格/allocator审计仍在caller，随后复制shape/stride、保留Storage并创建任务。
缓存命中不会省掉这些工作。每forward还创建25个完成事件，并最终record/wait/destroy后返还信用。

**协调工作随线程数明显增加。** MPSC4首轮累计CPU时间约为off的5.19倍，
自愿切换约为18.25倍；更多worker增加CPU与调度工作，却没有降低forward延迟。
入口CAS之外仍有core/state、计数、allocator和原PTA队列的锁及CV通知，GIL关闭不会移除这些C++同步。
平均用核没有吃满8核，因此不能仅归咎于CPU核数不足。

**跨阶段的保守同步仍在。** 每forward有18次fallback：Addmm6、BMM1、Cat2、Index1、Zero1、
Relu5、Sigmoid1及空Embedding1。每次均drain队列并同步默认流；Interaction仍走这条路径。
当前并行范围是host Prepare，NPU仍按原默认流FIFO执行。同步次数是源码/计数事实，
不能据次数断言它占了全部额外延迟。

## 同二进制独立阶段诊断

外部perf/uprobes保留原SDK解析与二进制，不使用SDK替换或Python profiler。
以native Snapshot第2次return至第3次entry划定热态100-forward区间。
两组均满足2500 Submit/Prepare/每类event及1800 fallback，零LOST、零未配对，数值和退役检查通过。

| 插桩区间，ms/forward | mutex4 | mpsc4 |
| --- | ---: | ---: |
| 诊断ROI平均 | 3.721 | 4.944 |
| caller的SubmitImpl累计 | 0.962 | 2.185 |
| Prepare，跨worker累计 | 0.298 | 0.298 |
| fallback Admit累计 | 0.430 | 0.378 |
| retirement event wait累计 | 0.856 | 0.678 |

Submit增量约1.223ms，而Prepare累计区间几乎不变，支持新增成本集中在提交/发布/确认路径的归因。
MPSC事件record/wait/destroy中94.24%在release线程，5.76%在caller；不能说consumer逐op全部同步串行。
这些区间有嵌套和跨线程重叠，且探针本身有开销，**不能相加成未插桩E2E的阶段百分比**。

## 后续优化重点与证据范围

优先考虑缓存/成本感知的热路径选择、单caller的直接发布路径、减少可证明冗余的fallback屏障、
批量异步retirement及定向唤醒。每项都需要重新证明保序、所有权和设备完成边界，
不能直接删除同步或把所有算子无条件送入更多worker。本报告未实现这些后续方案。

资源监控以约4秒周期采样。十个性能进程区间均未观察到14/15号卡foreign进程，
容器和UDA归属正常；早期正确性有共享/未覆盖区间，未声明整个宿主机独占。
独立插桩诊断不混入上述10个性能窗口。测试已完成并主动停止自己的测试容器，释放14号卡。

机器可读证据：

- [性能统计与环境](evidence/performance_pilot_20261115.json)
- [30000个未插桩原始时延样本CSV.gz](evidence/performance_samples_20261115.csv.gz)
- [独立插桩计数与区间](evidence/instrumented_profiles_20261115.json)

候选SHA256：`d8943e56a5b234b2dc7cb0091861087a611437661b0ddd6479d5ca610a7ccaa4`。
代码来源e53466b，报告工作树以107d977为基础；其他未提交优化不属于本次测量。
`global_lock_free=false`，当前仅入口primitive无锁。

私有原始证据归档110项、13219576字节，SHA256
`572f5ad2085c41c8a07623eaaf1aa80de0b79153ec91bfc41cd12980ec3b1238`。
包含全部原始worker结果、监控、私有诊断脚本、探针manifest和perf原始记录；
公共CSV.gz与JSON可独立复算上述统计，私有环境路径和SDK二进制不上传公共仓库。
