# R32：真实 A3 上的空输入修复与五组对照

空输入问题已修复，原模型和 R31 的 MetaSyn26 混合输入保持不变。真实多流 E/EI 及选择性 E/EI 的数值、路由和生命周期检查均通过，但本次 E2E 没有收益。四种调度方案在全部 10 轮中都慢于 off。

## 实测结果

| 方案 | E2E 中位数 ms | p95 ms | 同轮配对耗时变化 | 快于 off 的轮次 |
| --- | ---: | ---: | ---: | ---: |
| off | 2.181525 | 2.218641 | — | — |
| E | 3.786057 | 3.937966 | +73.27% | 0/10 |
| EI | 4.020904 | 4.201622 | +83.77% | 0/10 |
| 选择性 E | 2.670566 | 2.787650 | +22.62% | 0/10 |
| 选择性 EI | 2.943003 | 3.075569 | +35.25% | 0/10 |

表中延迟是 10 个窗口各自中位数/p95 再取中位数；百分比是同轮 `方案/off - 1` 再取中位数，因此不等同于直接对表中两列相除。原始轮次和统计口径见 [结构化结果](R32_REAL_A3_RESULTS.json)。

off 使用同一修复包并关闭调度；E 为全部 Embedding 多流；EI 再调度原始 Interaction 的两次 Cat、一次 BMM、一次 Index。选择性组使用事先固定的元数据平均 L≥64 阈值：原 26 张表中仅第 2、9 张表走两个 producer 流，24 张小表留在 caller。模型中完全为空的表完整保留。

## 空输入根因与修复

完全为空的 SUM EmbeddingBag 在该 CANN 路径中可能不写输出内存。allocator 留下的旧值会进入后续 Interaction。修复位于 `patches/eager_parallel_r32/op-plugin/op_plugin/ops/opapi/EmbeddingBagNpuOpApi.cpp`，同时覆盖 `_embedding_bag` 和 `_embedding_bag_forward_only`：在全空 SUM 时用原生 `at::zeros` 初始化输出，随后继续原来的 CANN 调用。原有参数校验、辅助输出、非空计算和 E/EI 入口都保留。

在正式性能测试结束后，额外构造了同版本反证包：复用全部已编译对象和相同 Python 绑定，仅将这一个算子对象还原为 pinned upstream 源码并重新链接。未修复包的 72 个全空案例全部失败，72 个非空案例全部通过；换回正式修复包后，144 个案例再次全部通过。用例覆盖两个入口、int32/int64、include_last_offset 开关、B=1/3/128 和三次 allocator poison。反证进程确认实际加载的是独立未修复原生库，Torch git 与 GIL 状态相同。正式包源码和二进制未变。

修复源码 SHA256：`cf6e5d224bd030b0974dfb62e82a940e68884e3f0f7422b88179715b333b3975`。
未修复 upstream LF SHA256：`fb5b7f48d5a1f2ac85f7c10faab9410171a6d809c62a329af9b5915ef9648a6f`。

## 正确性与实际调度证据

55 个全新 worker 全部成功退出，包括先执行的五组正确性和 50 个正式窗口。每个进程额外检查八次完整 forward、全部 26 个 Embedding 输出、Interaction 和最终输出。相对原模型 CPU 参考，最大绝对误差：Embedding `0.0`，Interaction `1.7881393432617188e-07`，最终输出 `8.940696716308594e-08`。数值检查和 CPU/H2D 准备均在计时之外。

原生快照为 `native_immediate`，实际流 ID 为 caller 0、producer 96/97、consumer 98；E/EI/选择性组的每次 forward 路由数量、事件依赖和关闭门禁均通过，未出现意外回退、错误、隔离状态或遗留 owner。host_workers 为 0：本实现是在原生调用中选择真实 NPU 流。映射确认加载真实 CANN；该版本优先加载 `libopapi_nn.so`、`libopapi_math.so` 等拆分库。37 项源文件/二进制 pins 在每个 worker 前后复核。

通用 fallback 计数包含正常的 caller 路径：E/选择性 E 每次为 16（12 个 MLP 算子加 4 个 Interaction 算子），EI/选择性 EI 每次为 12 个 MLP 算子。全部窗口均与该预期一致；`policy_uncertain_fallback` 和 `small_dependency_drains` 均为零。

| 方案 | 进程 CPU ms/forward | 侧流 Embedding | 侧流 Interaction | 事件记录/forward | 事件等待/forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 2.219478 | 0 | 0 | 0 | 0 |
| E | 4.143511 | 26 | 0 | 27 | 4 |
| EI | 4.440653 | 26 | 4 | 31 | 8 |
| 选择性 E | 2.759965 | 2 | 0 | 4 | 4 |
| 选择性 EI | 3.091663 | 2 | 4 | 9 | 8 |

进程 CPU 值为每个窗口的进程累计 CPU 时间除以 3000，再对十轮取中位数，包含该进程各线程及计时循环开销，不能视为互斥的阶段耗时。数据表明跨流管理增加了 CPU 工作；选择性调度减少记录、保留和审计等管理量，但剩余成本仍未被当前小模型的可重叠计算抵消。本轮没有互斥阶段剖析或设备 timeline，不能据此精确分摊事件、allocator、元数据审计各自耗时，也不量化设备 kernel 的重叠比例。

## 固定负载与环境

- 原版 DLRM：B=128、26 张表、V=1024、D=16、FP32 unweighted SUM，原 MLP、Interaction 和 forward 不变。
- 输入仍是 R31 的小规模 MetaSyn26 投影，含 ragged/empty bags；大表平均 L 约 102.23/76.81。此处的 V=1024 投影和小权重规模是结论适用范围，负载定义见 [R31 报告](R31_SELECTIVE_META_HOSTSIM.md)。
- model SHA：`9d430a06b378ee02266a623d219c14c642b96283648831d2f998f251c244f5d5`。
- state SHA：`6d0485a04fac556bf636f7df7541d2bf6bf4c3cb7b4641149c1f73fc7532cd19`。
- input SHA：`c4507e2138c2a5ee730e3c8b3e279b65ee2464d6a6fe4cbe3c514adc3c795ec1`。
- dense SHA：`352958c0d5d686e9a66533b6c7d38acb0440f9dc218be715b9c3e540b71945f9`。
- 真实 A3 `Ascend910_9362`，物理设备 14，容器内逻辑设备 0。选定的驻留容器预检仅有 sleep、没有计算进程；主机还承载其他任务。设备 PCI NUMA 信息为 -1、local_cpulist=0-319，本次固定主线程 CPU affinity=8..15。
- Python 3.14.6t、自然 GIL=false、Torch `2.13.0+cpu` / git `cf30153c4c131c8164ee7798e5022d810682e2cb`、NumPy `2.4.4`、CANN 9.1.0、驱动 25.5.1；Torch intra/inter-op 各 1，TASK_QUEUE_ENABLE=0、PER_STREAM_QUEUE=0、ASCEND_LAUNCH_BLOCKING=0。
- 私有 glibc 2.39 ELF loader 提供解释器所需 ABI；旧 Docker 的 clone3 EPERM 使用进程内附加 ENOSYS 过滤器，使 glibc 回落至原已允许的 clone。没有更改宿主库、驱动或全局 Docker 配置。运行时依赖为完整真实依赖，原模型直接导入。
- 实际构建 `TORCH_NPU_EXPERIMENTAL_FREE_THREADED=ON`、`TORCH_NPU_EAGER_PREPARE_HOST_SIM=OFF`，真实 CANN 头/库，新编译原生核心、ATB 辅助库和 `_C` 绑定。CANN 头文件布局映射和可选 AML 声明头在编译回执中记录；构建和运行环境的 CANN 核心库 SHA 一致。

## 测量与复核

五个正确性进程先全部通过，然后执行十轮五组 Williams 顺序；前向/反向各五个序列，每个位置和有向相邻 pair 均出现两次。每组每轮 50 warmup、3000 timed，共 150000 个正式样本。采样区间为原始 `model(*inputs)` 至 `torch.npu.synchronize()` 返回；模型加载、输入构造/H2D、快照、额外数值校验和进程启动排除。所有 worker 起止区间串行、不重叠。

首个预检 cohort 在正式计时前因测试工具只识别 monolithic opapi 名称而停止；核对实际映射和 pinned loader 后修正测试工具。完整性能来自随后单独冻结的 campaign002，不混用失败 cohort。正式期间不再修改模型、输入、原生库或计时 worker。

独立标准库审计不导入 worker/controller，直接复算全部 55 个结果、150000 个原始时长、数值/路由/关闭门禁、身份与顺序平衡。归档下载后在本地再次校验所有成员并复算一致。本地既有回归 175 项，170 通过、5 个可选原始源码路径集成项跳过；真实 A3 144 项空输入回归及五组全部数值门禁均实际执行。

## 代码、包与证据

实现基于 R31 checkpoint `a70baf10cd599188b2a3e25249a5b907ed0e5279`。生产修复是 `patches/eager_parallel_r32`；真实设备测试工具是 `benchmarks/eager_parallel_r32`。

- 正式 `libtorch_npu.so` SHA：`90196668f98f55945a8910979ac8ae5e0619f6bfd68d257b0f803bacbb43a11e`。
- 正式 `_C` SHA：`7da52be0690c8af88e17497b5a8cf9620b21f42c5398f9df739ae043895ec134`。
- 未修复同版本控制库 SHA：`a8a72264e853ec571f9b041904d11f1d28132847673f7a7a0abfeffcfcb319c6`。
- 证据包 `r32-real-a3-evidence-001.zip`：1771899 bytes，181 个带 size/SHA 的 payload 加 manifest，SHA `fa8f2d5d66642ac22a3985303f5642e3df8c504d19ec845fbc04b68e5742c607`。包含全部原始样本、源文件、配置、编译/控制回执和生命周期；不包含大二进制。

现场与包保留在 A3 的 `/home/zyz/r32-a3-20261111`。实际运行包位于选定设备容器内同名目录的 `attempt-real-001/packages/torch_npu`，完整启动参数保留在宿主该目录的 `candidate-launch-config-idle14.json`。本地证据副本在项目根 `.cache/r32-transfer/`。原生全局 qualification flags 保持实验状态；本报告给出该固定负载的真实设备证据。
