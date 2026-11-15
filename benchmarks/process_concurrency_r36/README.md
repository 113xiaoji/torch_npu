# 多进程完整 forward 对照

[CPU与真实A3结果](RESULTS.md)包含三轮吞吐、延迟、内存及长尾分析。

多个全新 exec 进程，各运行一个完整原版 DLRM，请求与 R35 完整 forward 场景一致。
每进程独立模型、输入存储、PTA 上下文和 NPU stream；共享同一张 NPU，均固定 CPU8–15、
Torch intra/inter-op=1。没有跨请求合批，每请求 batch128，空表和 ragged 输入保留。

请求全局编号0..2999，进程rank执行rank::进程数，输入bank编号始终为id%8。
预热每进程24次；所有进程ready后，controller发布200ms之后的共同起点。
所有进程完成设备请求后，才统一允许输出复制、数值校验与权重hash。
吞吐为总完成数除以共同起点至最后响应的窗口，不能把每进程吞吐直接相加。

两种运行时必须分开：

- `sameft-gilon`：同一Python3.14.6t/Torch2.13二进制，明确`-X gil=1`且前后验证GIL开启。
  这是隔离GIL与进程效应的控制，仍保留FT构建，不能称作标准非FT。
- `standard-gil`：普通Python3.12.13/Torch2.10/PTA2.10，`Py_GIL_DISABLED=None`。
  版本与FT环境不同，只能作为额外参考。NPU沿用已审核、字节固定的标准2.10空表原生修复；
  模型和输入不改，修复库与provider映射均记录。

`worker.py`校验解释器/框架/模型/输入/源码/原生文件身份，保存每请求原始时刻、线程CPU和
数值校验；FT原生队列保持关闭，并验证关闭ACK。`controller.py`只使用标准库，启动独立
进程组，控制同步和失败回收。`contracts.py`从原始请求重算吞吐/延迟，拒绝缺失、重复、
错误分片、错误数值或跨rank身份混合的报告。

在预配置的容器内运行：

```text
python -m unittest discover -s test/process_concurrency_r36 -v
python benchmarks/process_concurrency_r36/controller.py --launch LAUNCH.json --config CONFIG.json --worker ABS_WORKER.py --world-size 4 --total-requests 3000 --warmup 24 --timeout 220 --output NEW_DIRECTORY
```

launch的`argv_prefix`包含环境、可选loader、真正解释器与可选`-X gil=1`，禁止在导入Torch后
fork。config列出`cpus`、`runtime_kind`、`expected_runtime`、解释器及源码/模型/provider pins，
`frozen_r35_source_root`指向冻结的`runner/benchmarks`。标准运行时的
`standard_empty_fix`必须同时出现在pins中且符合已审核SHA。

PSS/RSS使用all-ready之后、发go之前的近同时顺序快照。PSS可按进程求和以减少共享库
重复计数；它是驻留内存快照，不是峰值，worker各自after-memory不能混加作为并发总量。
首请求晚调度被计入共同窗口；done标记写盘/轮询有少量host扰动。
输入创建、H2D、参考值、进程启动和输出校验均不计入请求窗口，网络/预处理/序列化未模拟。
独立stream和host区间重叠不证明NPU kernel重叠。
