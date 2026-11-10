# R30：原版 DLRM 各表混合 L 的端到端结果

**本次混合 L 输入没有获得收益。E 配对中位延迟增加41.63%，EI增加42.05%；
两者均为12/12轮更慢。** 26个EmbeddingBag输出、Interaction和最终输出的最大绝对误差均为0.0。
这是原版DLRM在既定CPU数值host-sim上的结果。

## 结果

| 模式 | 中位延迟 | 配对延迟变化 | p95 | 配对p95变化 | CPU时间/forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 1.216093 ms | 基线 | 1.233810 ms | 基线 | 2.035420 ms |
| E，只调度Embedding | 1.723990 ms | **+41.63%** | 1.741180 ms | +41.14% | 2.779041 ms |
| EI，调度Embedding和Interaction | 1.728938 ms | **+42.05%** | 1.748200 ms | +41.53% | 2.895360 ms |

延迟/p95列分别是12个进程内median/nearest-rank p95的跨轮中位数。
百分比是逐轮`candidate/off−1`的中位数，不能直接用表中汇总值相除替代。
E逐轮延迟增加39.99%–43.47%，EI增加40.97%–42.77%，均为0/12轮更快。
CPU时间是计时窗口内全进程所有线程CPU时间除以样本数，再取跨轮中位数；
它不是各阶段互斥耗时分解。本轮并行同时增加了延迟和CPU工作量。

## 输入与模型

采用公开MLPerf DLRMv2配置的26表长度向量：

```text
3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1
```

[向量来源](https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/dlrm_dcnv2/implementations/b200_preview_merlin_hugectr/train.py#L55-L82)。
本轮将这个长度向量用于已有原版DLRM，保持原模型源码/forward、FP32权重、dense输入、
batch128、26张表、每表1024行、Embedding维度16、sum pooling和MLP/Interaction结构不变。
模型源码SHA256仍为9d430a06…，权重6d0485a0…，dense输入352958c0…。

每个样本共查214个ID，表间平均L约8.23；24张表L≤12，另外两张为27和100。
每表在128个样本中使用同一个固定L，offset步长严格为该表L，索引数为128×L。
各表结果仍为`[128,16]`，Interaction为`[128,367]`，最终为`[128,1]`。

ID取自冻结R29 L256输入的每样本前缀，无新增随机抽样、无去重。
L1保留原始两个ID中的第一个，L≥2保留原始两个ID；原输入没有被修改。
完整混合输入SHA256为
`40d6a16803f328885b66357bbcfdb680eb5578b15a641a61421069e4755be511`，
其L256基底为`9f9f12a6474c315fff345d330ce6be1679c936a1475cd9916d252d2a1cd81782`。
所有arm使用字节相同的输入，root和独立实现者分别构造得到相同hash，204原模型集成再次确认。

这里模拟的是该长度配置下的原版DLRM；它没有复现完整DLRMv2模型、真实流量的ID分布，
也没有模拟同一表不同请求之间的变长分布。表权重总量仍约1.625MiB。

## 为什么没有收益

与此前所有表统一L64以上的实验相比，这次只有少数表查表量较大。
统一L64每样本查1664个ID，而本轮是214个；这里只比较索引数量，不按比例推算耗时。
与此同时，每forward依然执行42个数值kernel，其中26个为EmbeddingBag。

| 每forward | off | E | EI |
| --- | ---: | ---: | ---: |
| 数值kernel | 42 | 42 | 42 |
| SDK队列任务 | 42 | 229 | 272 |
| SDK event record，含allocator bookkeeping | 0 | 183 | 222 |
| SDK event wait | 0 | 4 | 8 |

因此，较少的查表工作仍需承担相同次数的调度、事件和allocator管理。
本次结果与“小算子工作不足以覆盖并行管理开销”的解释一致；本轮没有做互斥阶段计时，
不能把全部额外延迟精确归因到某一个同步点。

保持现有轮询分配后，两producer各执行13张表，但每样本ID量分别为138和76；
零起始第20张表的L100占全部ID量约46.73%。这证明了查表量不均衡，不能直接把ID量当作
线程执行时间，也不能据此断言负载不均衡贡献了多少延迟。本轮没有调整调度策略。

## 测量与验证

运行于192.168.41.204，全部新文件在`/home/zyz/r30-mixed-l-20261110`，
复用`/home/zyz/r28-multihot-20261110`的已资格化环境与二进制。
同boot aeefacdc…、Python3.14.6t自然GIL=false、Torch2.13 CPU git cf30153…、NumPy2.4.4，
同native64cbe0…及SDK007。没有修改或重编native、SDK、模型或运行时。
固定8个物理核144/146/148/150/152/154/156/158、NUMA1和单LLC；
TASK_QUEUE_ENABLE=0，Torch intra/inter-op各1线程。

- 3个cold进程先资格化，随后36个perf进程；39个PID各不相同，进程起止区间串行无重叠。
- 每arm12轮，每轮50warmup/3000timed，共108000个正式计时forward；六种arm排列各两次。
- 计时边界仍为原`model(*inputs)`到同步返回。外部输入构造、模型加载、H2D、快照和
  额外数值校验排除；forward原有Python索引构造仍计入。
- 额外校验26个bag、Interaction及最终输出；均最大绝对误差0.0，检查和临时张量清理在计时外。
- warmup/timed/额外stage分别重算数值kernel及原生调用增量，确认两producer的13/13次查表、
  EI独立消费流和caller MLP；执行了实际E/EI调度。
- 所有进程exit0，无timeout，进程组关闭；native closure和SDK完成/对象/错误门禁全部通过。
- 计时窗口3.646–5.225秒。共享主机；预检1秒样本中所选核均空闲，运行plan也保留负载记录。
  不宣称主机独占，也未触碰其他任务。
- 本地124项测试120通过、4项可选原始源码路径跳过；204本轮原模型CPU集成10项全过、无跳过。
- 实施前独立审查和冷启动原始回执审计通过，计时完成后再次核对全部回执。

## 证据

controller SHA256 `d5d00bd4527c23b50dbc9589d675e07e8c27031455badbeb2fb417c1e885f521`；
workload SHA256 `8c058e4a9cfadaf5cd42083a1a0e135372a93568b7537b66e59989f3290db23a`。
R27worker、R28/R29workload源码字节未变。

`r30-evidence-final-001.zip`含198个SHA/size标记的payload成员和manifest，979006字节，SHA256：
`8d3b8ff9e542a6a8c9027900cd0b259fc09c5688509bf5897272cb856843c909`。
归档留在204任务root；本地副本位于项目根`.cache/r30-transfer/`，所有归档成员均已校验。
主要目录为`preflight-001/`、`runner-001/`、`cold-001/`、`perf-001/`。
独立标准库审计不导入worker/controller，重算39个进程的身份、调用计数、流、数值、
生命周期和全部108000个计时样本；root另一条统计路径重算结果一致。
原始日志、环境二进制和私有账本未上库。

本轮不能将[R29高L的收益](R29_HIGH_L_HOSTSIM.md)推广到该混合配置。
所有性能结论均限于本次CPU数值host-sim，真实NPU尚未验证。
