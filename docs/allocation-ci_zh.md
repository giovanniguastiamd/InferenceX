# 使用现有资源分配的 CI

[English](allocation-ci.md) | **中文**

分支中的 `allocation-benchmark.yml` 使用现有矩阵生成器选择一个 GLM-5.2 FP4
SGLang AgentX 配置（TP4/EP4、HiCache、并发数 4）。可复用的 benchmark 工作流和
配方仍是执行依据。

提供一个正在运行、独占八张 GPU 的单节点 Spur 作业。在该节点注册临时 runner，
名称为 `mi355x-amds_<unique-id>`，参数为
`--no-default-labels --labels allocation-<node>-<job-id> --ephemeral`。
不要添加通用资源池标签。每次注册只接收一个工作流作业，重复测量需要重新注册。

默认只验证所有权、节点、资源分配、缓存访问权限和模型分片是否存在，不启动 Docker，
也不代表已建立基线。记录 `.hf-revision` 标记，但该标记不能证明所有权重都对应此版本。
接受基线测量前仍需完整的来源信息和服务端实际配置验证。

在工作流进入默认分支前，可在 `tuning/allocation-ci` 分支提交明确的
`.github/allocation-request.json`，用 `job-id`、`node` 和布尔值 `validate-only`
选择运行。该文件变更会触发该分支的 CI，不要复用已过期的作业 ID。其他分支不受
此 push 触发器影响。工作流进入默认分支后也支持手动触发。

此执行路径跳过通用的资源清理和 git 修复。`utils/allocation_guard.py` 在启动前以及
容器运行期间每十秒检查所有权，不申请也不取消资源分配。失败或取消时只删除本次运行
唯一命名的容器。如果已有容器正在运行，则拒绝 benchmark，不删除已有容器。

权重只读挂载到 `/model`，调用者的持久化 HF 缓存挂载到 `/hf`。容器使用操作者的
UID/GID 和明确的 GPU 设备，不使用 privileged 模式。资源分配证据单独上传；配方产物
沿用可复用工作流的标准上传路径。

Runner 生命周期监督器必须在资源分配结束时停止监听进程并移除注册，包括从未接收
作业的空闲 runner。容器检查不能替代该监督器。宣布集成完成前，必须验证资源分配
丢失、runner 终止、产物上传以及镜像的非 root 运行兼容性。
