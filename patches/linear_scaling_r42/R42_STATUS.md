# R42 workspace 资格查询实验

本轮将 workspace 复用路径中的 allocator 池状态查询改为可选的原子读取。
源码已完成 ARM 构建、CPU 并发验证和两种模式的真实算子资格检查。
完整模型检查及端到端性能对照仍在进行，尚无 R42 吞吐增益结论。
四线程接近线性增长的目标尚未达到。

前一轮 R41 的完整 A3 对照已独立审计：组合优化四调用者配对增益
中位数为 8.41%，四/一扩展配对均值为 2.75 倍。
见 [R41 完整结果](https://github.com/113xiaoji/torch_npu/blob/codex/r41-linear-scaling-20261116/patches/linear_scaling_r41/R41_RESULTS.md)。
本分支从 R41 源码提交分出，既有 R41 准备状态文件是历史记录。

## 变更与使用

`TORCH_NPU_WORKSPACE_CACHE=1` 启用既有 R41 workspace 复用后，选择：

| `TORCH_NPU_WORKSPACE_POOL_QUERY` | 查询行为 |
|---|---|
| 未设置或 `locked` | 使用既有 allocator 查询锁，默认模式 |
| `atomic` | acquire 读取独立缓存行上的池活动状态 |
| 空字符串或其他值 | 拒绝执行该资格查询 |

模式在进程内缓存；比较不同模式须使用新进程。原生导出
`OpApiWorkspacePoolQueryR42()` 返回实际模式，测试同时核对环境、配置和 getter。

`beginAllocateToPool` 和 `endAllocateToPool` 在原有互斥锁内修改池集合，
随后 release 发布活动标志。分配、释放和池切换继续使用原锁。
自定义 allocator 无可核验状态时，原子查询返回活动并绕过复用。
既有 capture、外部 stream、异常资源及资源 epoch 资格限制保持生效。

状态标志用于观察资格。调用方仍须正确安排池进入、退出和操作的先后关系。
本次已验证同线程及 `threading.Event` 建立顺序的跨线程交接；
无序的并发池切换、完整 DLRM 图捕获及自定义 allocator 场景尚未资格确认。
默认模式保持 `locked`。

## 已完成验证

- 实际生产池 begin/end 方法的嵌套、异常及 80,000 次并发 scope 检查。
- 未设置、locked、atomic、空值和错误值五种模式检查。
- 两模式均通过 R41 allocator、私有池、unsafe、初始化及 epoch 准入检查。
- ARM 增量构建重新编译两处受影响实现，复用冻结 R41 的其余对象；
  R41 原库保持原字节。R42 原生 SHA256：
  `0dc493a1cd8b36a98d30482df55a3c8dce0baf1ca9edc8fc5dd8e2a58bf35753`。
- 两模式各验证 88 个输出，共 176 个。使用原模型最大非空表的
  EmbeddingBag、原权重及八种原输入；先证明 workspace 正常命中，再检查
  三种真实私有池选项、保序跨线程 scope、图捕获绕过及八次回放。
  私有 segment 的 pool ID、进入期间绕过、退出后命中恢复及数值均通过。

算子检查中的图使用固定原 bank0，其余 scope 检查覆盖原 bank8。
这部分是正确性与资格证据。短阶段仅有相交的完整资源查询区间，
不证明持续独占，也不提供端到端性能结果。

原始算子证据包 `r42-scope-proof-c09acb06-evidence.tar.gz` SHA256：

`a7546848378fa84a0b764bf6461bc539de7a6c42135b1a3de0512c29bc3ffb12`

## 后续对照

保持原 DLRM、权重、八种输入、batch128、自然 GIL-off CPython3.14.6t、
CPU8..15、Torch1/1、PSQ1 和私有 stream，比较冻结 R41 AB、R42 locked、
R42 atomic 在 1/2/4 调用者下的结果。完整三模型检查和九组 DLRM 检查
必须先通过；性能部分需重新采集 27 个窗口，每窗口 3000 请求，
每调用者预热 24 次。仅复用强绑定、完整封存的正确性阶段，旧性能复用为零。

每次启动重新检查双芯片 proc/usage、UDA 和 fuser；出现外部任务则清理
本次自有资源并停止。性能接受仍使用完整原 ROI 资源门槛，
不会采用短正确性阶段的查询区间相交口径。

独立完整结果发布前，保留本实验为显式选项；验收同时比较单线程吞吐、
四线程绝对吞吐、各阶段耗时和 p99。
