# R31：按平均 L 选择 Embedding 并行路径

实现了模型代码之外的原生选择性调度：达到阈值的EmbeddingBag走已有producer流，
独立的小调用执行原来的caller流body。默认阈值0保持原E/EI行为；本实验固定为64。
当前尚未得到自适应路径的正式端到端性能结果，不能声称已经获得收益。

## 调度与依赖

- 使用`indices.numel()/bags`作为一次调用的平均L，其中`bags`依据offsets长度和
  `include_last_offset`计算。它是整次调用的粒度，不拆分同一batch内各bag。
- 小调用保留线程归属、环境、推理和元数据检查；不做成功路径的输出visitor、
  owner注册或每小调用事件。输入与待完成side输出的原生Storage字节区间重叠时，
  先保守drain；无法确认安全的分配或元数据也回到有完成栅栏的原执行路径。
- caller提交小调用之前标记frontier为dirty。随后side生产或消费会等待新记录的caller
  frontier，保证读到此前的小调用结果。每次刷新使用不同事件，完成证明前不复用句柄。
- 选择性模式预留`2*capacity+1`事件；容量耗尽前先完成现有group。未知算子、变更和
  观察边界继续使用原有栅栏。仅有小调用时，显式close仍等待caller完成。
- 异常保留原始错误；失败恢复不能证明完成时，已捕获的输入及结果引用随domain隔离保留。
  恢复覆盖全部四条流，避免将仍被使用的内存提前释放。
- EI在没有side group时将Interaction留在caller。模型源码、forward与CANN算子没有修改。

配置示例，模型和输入应先放到`npu:0`完成初始化：

```python
import torch
from torch_npu import _eager_parallel as ep

with torch.inference_mode(), ep.scope(mode="e", min_embedding_bag_length=64):
    output = model(*inputs)
    torch.npu.synchronize()
```

## MetaSyn 投影

使用[Meta官方公开合成trace](https://github.com/facebookresearch/dlrm_datasets)的2021 base数据，
已校验2398183998字节和官方LFS SHA256。全856表平均L约15.811706，表级平均L最大约193.053482。
以全batch平均L排序后，取26个中点分位点，并用固定seed20261110打散到原模型表位置；
每表使用原始前128个bag，ID经过无损int64校验后模1024映射。

保留了ragged和空输入。投影共50850个ID，平均L约15.279447；模型表2/9的平均L分别为
102.2265625/76.8125，其余24张表低于64，其中表1完全为空。这里保持原版DLRM的26表、
batch128、词表1024、维度16、FP32权重、dense输入和网络结构；它是该模型规模下的MetaSyn
输入投影，不代表完整Meta生产模型或真实流量的性能。

## 已完成的验证与待测部分

已完成原生库全量编译和Python绑定编译，源码及原有SDK007/运行时输入均校验未漂移。
Python3.14.6t自然GIL=false下，新包四种原生配置/关闭的smoke通过。
本地175项相关测试170通过、5项因可选原始源码路径跳过；204上的本轮原始模型CPU集成
14项全过、无跳过。原模型源码/state/dense及完整新输入SHA均匹配固定值。

Linux已实际编译并执行18项原生测试：Release、ASan、TSan各6项，另有2项真实ATen探针通过。
它们覆盖协议、别名、容量/事件代际、输出生命周期、异常隔离和跨线程清理。
生产源码8文件在验证前后与编译输入一致。这些证据不构成真实CANN/NPU并发认证。

计划在同一新candidate上比较off、全量E、全量EI、选择性E、选择性EI。五组cold先通过，
再做10轮Williams顺序，每组50预热/3000计时，共150000个正式计时forward。
将保留逐轮median/p95、CPU时间、实际路由、事件/队列工作量、数值及进程回收回执。

当前204内网SSH在cold过程中断开，随后204/55连接均超时。已观察到cold的off和全量E通过；
其余cold的远端状态尚未回收，正式perf没有启动。恢复连接后须先核查旧controller和
`cold-001/complete.json`或`error.json`，再使用独立于SSH输出通道的文件日志supervisor继续。

代码位于`benchmarks/eager_parallel_r31/`，真实数据、构建和回执均保留在204的
`/home/zyz/r31-selective-meta-20261110`。结果尚待完成，当前不作加速结论。
