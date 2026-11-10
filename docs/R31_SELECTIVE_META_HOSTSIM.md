# R31：MetaSyn 投影与大 L 选择性并行结果

**选择性分流减少了全量并行的损失，但没有获得相对off的净收益。**
选择性E（AE）比全量E延迟降低19.01%，选择性EI（AEI）比全量EI降低19.31%；
相对off仍分别增加10.87%/11.37%，两者均为10/10轮更慢。
实际执行了每forward两次Embedding走producer、24次走原caller的路径，数值检查全部通过。

## 端到端结果

| 模式 | 中位延迟 | 相对off的配对延迟变化 | p95 | 相对off的配对p95变化 | CPU时间/forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 1.264665 ms | 基线 | 1.282430 ms | 基线 | 2.105435 ms |
| 全量E | 1.731148 ms | +36.86% | 1.748930 ms | +36.33% | 2.916292 ms |
| 全量EI | 1.741375 ms | +37.70% | 1.760865 ms | +37.30% | 3.035976 ms |
| 选择性E（AE） | 1.404040 ms | **+10.87%** | 1.421605 ms | +10.82% | 2.342375 ms |
| 选择性EI（AEI） | 1.407723 ms | **+11.37%** | 1.425325 ms | +11.17% | 2.447969 ms |

延迟/p95列分别为10个进程median/nearest-rank p95的跨轮中位数。
变化为逐轮`candidate/off−1`的中位数，不是表内汇总值直接相除。CPU时间为计时窗口内
全进程所有线程CPU时间/样本数，再取跨轮中位数，不是互斥阶段耗时分解。

AE逐轮相对off延迟增加10.15%–11.42%，AEI增加10.74%–11.75%；全量E/EI也均0/10轮更快。
直接配对全量并行基线时，AE相对E降低19.01%（10/10轮更快），AEI相对EI降低19.31%
（10/10轮更快）。这些是相对全量并行的改善，不能改写为相对off的加速。

## 输入来自公开 MetaSyn，原模型不变

使用[Meta公开合成trace](https://github.com/facebookresearch/dlrm_datasets)的2021 base文件，
固定revision `6f46ba7906e47a2f6bab6499056480d07d784eb8`，2398183998字节，官方LFS SHA256：
`1d4d3ca295fade795cbf973d9b29e721b7c76a2130684582d9f4ab579c95e772`。
全856表、65536样本的平均L约15.811706，表级平均L最大约193.053482、最小0。

在性能测量前冻结选择规则：按全batch平均L和源表编号排序，取26个中点分位点，
seed20261110打散到模型表位置；每表用原始前128个bag。源ID以float32存储，必须通过
int64往返无损检查，再模1024映射。保留原始变长bag、空bag及完全为空的模型表1。
分流阈值64来自之前独立高L实验，本轮不按结果调参或筛选源表。

投影共50850个ID/批，平均L15.2794471154。模型表2/9的平均L分别为102.2265625/76.8125，
共22917个ID；其余24张表低于64。阈值判断的是整次EmbeddingBag调用的平均L，
不是逐bag拆分；例如模型表18存在L200的个别bag，但平均43.25，仍留caller。

原始DLRM源码/forward、FP32权重、dense输入、B128、26表、V1024、D16、sum pooling、
MLP/Interaction结构不变。源码SHA仍9d430a06…，state6d0485a0…，dense352958c0…。
每表输出`[128,16]`、Interaction`[128,367]`、最终`[128,1]`保持不变。
这保留了原模型规模，不能代表完整Meta生产模型、词表规模或私有流量；表权重仍约1.625MiB。

## 确认走了选择性实际执行路径

五组共享同一个新编译candidate、SDK007、运行时和输入；只有原生策略不同：

| 每forward | off | E | EI | AE | AEI |
| --- | ---: | ---: | ---: | ---: | ---: |
| producer流Embedding次数 | 0 | 26（13/13） | 26（13/13） | 2（1/1） | 2（1/1） |
| caller流Embedding次数 | 26 | 0 | 0 | 24 | 24 |
| consumer流Interaction次数 | 0 | 0 | 4 | 0 | 4 |
| 数值kernel总数 | 42 | 42 | 42 | 42 | 42 |
| SDK队列任务 | 42 | 229 | 272 | 62 | 106 |
| SDK event record，含allocator bookkeeping | 0 | 183 | 222 | 16 | 56 |
| SDK event wait | 0 | 4 | 8 | 4 | 8 |

所有轮次的上述每forward计数一致；warmup、timed和额外stage分别重算。
选择性路径的`policy_uncertain_fallback`和`small_dependency_drains`增量均为0，
没有靠未预期回退或drain绕过选择性执行。MLP保持caller，AE的Interaction保持caller，
AEI使用独立consumer。各阶段验证、SDK完成及native关闭/owner/event退休门禁均通过。

小调用避免了成功路径的输出visitor、owner注册和完成事件，但还保留三个输入句柄、
元数据/环境检查以及Storage区间依赖审计。大调用仍需要原来的跨流保留和事件。
本次队列/事件数量明显减少，CPU时间也比全量并行下降，但仍高于off。
结果与“剩余管理成本尚未被两张大表的可重叠工作覆盖”的解释一致。
本轮没有做互斥阶段计时，不能将全部额外延迟精确归因到某一个检查或同步点。

## 正确性、资格化与实验方法

- 新原生库全量编译1127个目标，随后Python绑定编译通过；没有复用旧TorchNPU对象。
  13个固定DVM vendor对象作为输入单独记录。编译源码和原基线SDK/运行时pins在结束时再次核验。
- Linux Release/ASan/TSan各6项，共18项实际编译运行通过，另有2项真实ATen探针通过。
  本地175项测试170通过、5项可选原始源码路径跳过；204原模型CPU集成14项全过、无跳过。
- 同boot aeefacdc…、Python3.14.6t自然GIL=false、Torch2.13 CPU git cf30153…、NumPy2.4.4，
  同SDK007。模型和CANN算子未修改。固定8个物理核144..158偶数、NUMA1/单LLC，
  TASK_QUEUE_ENABLE=0，Torch intra/inter-op各1线程。
- 5个cold先通过，再50个perf；55个不同PID、均exit0/无timeout、owned进程组回收，起止区间无重叠。
  cold与perf之间发生过SSH中断；恢复后确认完整cold、旧controller退出、boot及pins一致，才启动perf。
- 10轮五组Williams顺序，前向及反向各5个序列；每个位置及有向相邻pair各出现两次。
  每组每轮50warmup/3000timed，共150000个正式样本，窗口3.791–5.266秒。
- 计时仍是原`model(*inputs)`至同步返回，包含原forward内部索引构造；外部输入准备、模型加载、
  H2D、快照和额外数值校验排除。每个进程检查26个bag、Interaction、最终输出，最大绝对误差均0.0。
- 共享主机，恢复后预检选定核1秒样本均空闲，运行plan也保留负载记录；不宣称独占或改动其他任务。
  正式perf使用独立于SSH的文件日志supervisor，全部完成后状态passed/exit0。

## 证据与范围

实现checkpoint `043bdee3a12eb8dd4000bccecaf412207dd85c00`。
新`libtorch_npu.so` SHA256 `3e75c2c03c7d957fa53cba60e7941f33e9e84db006eca1af14c836818e8e99fc`；
`_C`为`40ab3dead493713e913c38f48fafcb1d03c7e34f12dcd940fd28a93fd3b40d56`。
fixture为`5eb2663c2d00d64db83c9d5b858c78f80849841216b242c93cfcd72d2dfbade5`，
完整tensor输入为`c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1`。
controller047441e1…、worker0a7377b0…、workload6469fcfa…均与实际运行源码字节一致。

归档`r31-evidence-final-001.zip`含470个SHA/size标记的payload和manifest，2873886字节，SHA256：
`1617ea51a91c3a1aacb216246a656f7a970fb32fd14b00c4131bc18a03ef6ce9`。
包含全部原始样本、配置/命令/生命周期、实际源码、编译及sanitizer回执；所有成员逐项校验。
大数据、二进制及原始日志未上库。原始数据/运行现场仍在204：
`/home/zyz/r31-selective-meta-20261110`，本地副本在项目根`.cache/r31-transfer/`。

不导入worker/controller的独立标准库审计已重算55个进程、150000个样本、全阶段调用/流/数值/
关闭及身份门禁；root另一条原始统计路径得到一致结果。
**本结论限于该原模型规模下MetaSyn投影的CPU数值host-sim。真实SDK/NPU并发、带宽与设备timeline仍未认证。**
