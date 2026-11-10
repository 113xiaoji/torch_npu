# R28：只增加 DLRM bag 长度的 host-sim 实测

**L=2、8、16、32 四档均未获得端到端加速。** L=32 的负收益明显缩小，
但 E/EI 仍分别慢11.71%/12.62%，各档都是12/12轮慢于 off。

按用户后续指定，本轮在192.168.41.204执行，全部远端文件存放于
`/home/zyz/r28-multihot-20261110`。本轮每档都有新的 off 对照，未跨机器比较旧55耗时。

## 结果

| 实际 bag 长度 L | off 中位延迟 | E 中位延迟 | E 配对延迟变化 | EI 中位延迟 | EI 配对延迟变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.195500 ms | 1.696543 ms | +42.07% | 1.713953 ms | +43.86% |
| 8 | 1.192063 ms | 1.735978 ms | +45.27% | 1.750420 ms | +46.94% |
| 16 | 1.265953 ms | 1.783833 ms | +40.44% | 1.800785 ms | +42.08% |
| 32 | 1.530928 ms | 1.712780 ms | +11.71% | 1.724430 ms | +12.62% |

延迟列为12个进程 median 的中位数；百分比为逐轮 `candidate/off-1` 的中位数，
两者计算口径不同。每档每个模式36,000个计时样本；8组多流配对均为0/12轮更快。

| L | off p95 | E p95 | E 配对 p95 变化 | EI p95 | EI 配对 p95 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.216410 ms | 1.721795 ms | +41.24% | 1.741135 ms | +43.29% |
| 8 | 1.214765 ms | 1.752410 ms | +44.14% | 1.770375 ms | +45.96% |
| 16 | 1.285615 ms | 1.801020 ms | +39.96% | 1.827280 ms | +41.95% |
| 32 | 1.552255 ms | 1.730595 ms | +11.54% | 1.744165 ms | +12.48% |

p95列同样是进程内nearest-rank p95的跨轮中位数，没有混池不同进程或不同L的样本。

## 保持不变的内容

- 原始9文件来源、模型forward和运算结构保持原样；模型源码SHA仍为9d430a06…。
- FP32权重与初始化、batch128、26张表、每表1024行、Embedding维度16、sum pooling保持不变。
- bottom MLP `[13,64,32,16]`、top MLP `[367,64,32,1]`、dense输入和种子保持不变。
- 同一R27 native候选、SDK007、Python3.14.6t、Torch2.13 CPU、NumPy2.4.4。
- 8个物理核144,146,148,150,152,154,156,158，NUMA1、单LLC；Torch intra/inter-op各1线程。
- 三组均为TASK_QUEUE_ENABLE=0、单caller推理；进程自然GIL=false，没有强制PYTHON_GIL。

只在输入构造中改变每个bag的ID数量：每表ID总数从256增加到1024、2048、4096，
offsets步长分别为2、8、16、32。重复ID仍逐次参与求和，不去重。
每张表输出仍为`[128,16]`，Interaction仍为`[128,367]`，最终输出`[128,1]`。

L2的输入字节与R27原输入完全一致。更大的L保留每个bag原来的两个ID，再追加固定
seed20260919生成的每表128x30张量前缀；四档共享嵌套ID序列。同档off/E/EI的输入
hash完全相同，跨档dense/权重/model hash相同。

## 实验与正确性

- 12个cold进程先资格化，计时各自首次NPU forward；144个perf进程全部串行、新建进程。
- 每档12轮，每进程8次数值预检、50次预热、3000次计时，共432,000次timed forward。
- L顺序按轮做四档轮转，每档off/E/EI六种排列各运行两次。monotonic起止回执确认156个进程不重叠。
- 计时边界为原始`model(*inputs)`到`torch.npu.synchronize()`返回。模型加载、外部输入生成、
  H2D、快照和额外数值比较不计入；forward内部原有Python索引构造仍计入。
- 每进程另做一次不计时的完整分阶段检查：26个Embedding结果、Interaction及最终输出。
  所有实际误差最大值均为0.0；长bag的最终sigmoid饱和没有被用来掩盖中间错误。
- SDK调用增量仍为每forward Emb26/Cat2/BMM1/Index1/Addmm6/Relu5/Sigmoid1。
  warmup、timed和额外stage分别核对，未把stage混入计时计数。
- E/EI的两条producer每forward各执行13次Embedding；EI的Interaction确实在独立consumer。
- 156个不同PID均exit0、无timeout、进程组关闭；SDK无错误/注入故障/在途数值kernel，native owners退休。
- 144个计时窗口为3.559–5.457秒。主机共享使用，不宣称独占；尾延迟波动保留在原始样本内。

本地测试99项：97通过、2项缺本地原模型路径而跳过。204上原始模型集成测试7项全部通过、无跳过；
其四档原始loader检查使用独立子进程，遵守冻结loader禁止重复导入的约束。
输入生成、调度轮换、身份检查和数值饱和负例均有测试；没有修改或重编native/SDK。

## 能得出的结论

L32相对L2，off中位延迟从1.1955增长到1.5309ms，E/EI分别从1.6965/1.7140增长到
1.7128/1.7244ms；这一现象与更多查表工作被并行隐藏相符，但不是对独占阶段成本的直接测量。
L8/16也说明收益没有随L单调改善，不能把行数增16倍理解为可直接获得16倍收益。

协调任务的次数没有减少，四档均有以下统计：

| 每forward跨轮统计 | off | E | EI |
| --- | ---: | ---: | ---: |
| 实际数值kernel | 42 | 42 | 42 |
| SDK队列任务 | 42 | 229 | 272 |
| SDK event record（含allocator bookkeeping） | 0 | 183 | 222 |
| SDK event wait | 0 | 4 | 8 |

进程内所有线程CPU时间/forward的跨轮中位：L2为1.958/2.662/2.776ms，
L32为2.385/3.139/3.254ms（off/E/EI）。额外协调仍占成本，本轮范围内未跨过盈亏平衡。
没有把总差值全部归因到某一项，也不把调度器自身27/31个record与SDK总183/222混为一谈。

**这是CPU数值host-sim结果，不是CANN/NPU性能认证。** 该模型固定维度16和小词表，
Embedding表权重仍仅约1.625MiB；本轮未改变词表或维度，也未测试L64/128，不对这些负载推断收益。

## 环境和证据

原55的运行环境目录已不存在。按用户指定迁移到204后，使用55离线归档恢复了
字节一致的Python3.14.6t和Torch运行时；没有替换为其他已安装的Python3.14.3t环境。
native/SDK/原模型及依赖均复制到新任务目录并验证。主要路径：

```text
/home/zyz/r28-multihot-20261110/
  runtime-001/venv/bin/python
  assets-001/packages-ft/torch_npu
  assets-001/sdk-007
  assets-001/model-upstream
  runner-001/benchmarks/eager_parallel_r28/
  base-config.json
  preflight-204-001/
  cold-001/
  perf-001/
```

Python ELF SHA256为79db688405639c768cc9b975e98d21596a2040bd8f59496ce353d9a438be97cd；
native为64cbe0cbc95dee6280d8ad57555cf0365ff3a27014bbbad95dd4c9bf99b8226a。
运行环境归档SHA75d77550d851a66df2d5a9564078c71db35fc21b1188a55e91871f93bd672348。

最终`r28-evidence-final-001.zip`含666个有hash的成员和manifest，SHA256：
`5b023f22d8e6fbee7c38a52d4780713fb25816dc0a52e017944b339e2050ca75`。
本地副本在项目根`.cache/r28-transfer/`，原始日志、运行环境和二进制未提交到仓库。
独立标准库审计器未导入worker/controller，重新核对432,000个样本、配对统计、输入、
阶段数值、warmup/timed/stage计数、实际流、身份、命令、回执hash和全进程串行区间，全部通过。
