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

## 实际配置检查（M2e）

实际执行必须在请求或手动触发输入中提供 `image-digest: sha256:<64 hex digits>`。
提交前从目标镜像仓库确认摘要；标签或本地 Docker 镜像 ID 不足以证明镜像身份。
检查程序按摘要拉取镜像、检查 `RepoDigests`，使用解析后的镜像 ID 创建容器，
并在启动前核对容器的 `.Image`。

专用工作流明确请求 EAGLE、五个推测步骤和六个 draft tokens。六个 token 包含
验证 token，不能与五个 draft steps 混淆。服务就绪后、AgentX warmup 前，配方调用
`utils/verify_sglang.py`，读取 `/get_server_info` 和 `/v1/models`，比较服务模型名、
模型路径、TP/EP、推测算法、步骤和 draft-token 数量与调用者请求是否一致。
缺少字段、类型错误、端点不可用或值不一致均使作业失败。支持平铺字段和嵌套的
`server_args` 对象；不支持的 API 结构应在固定镜像中核实，不能猜测。

资源分配产物包含 `allocation-evidence.json`、`allocation-server-expected.json`，
以及检查开始后生成的 `allocation-server-validation.json`。验证报告仅保存选定字段、
请求值、观测值和失败详情；原始 API 响应可能包含凭据，因此不保存。
每次执行前删除旧报告。即使进程退出码为零，如果没有新的成功验证报告，检查仍不通过。
预检查失败或资源分配丢失时，证据文件仍记录运行身份和错误。

本地测试覆盖 API 比较和生命周期分支，不能证明固定镜像兼容性或已建立测量基线。
数据导入和看板集成仍待完成。文件来源按下文检查。

## 模型和数据集文件来源

专用工作流固定模型 revision `386bd0e4ec821f7b07975701cec3c3b953a5576a`，以及
[`campaign/glm52-model-manifest.json`](../campaign/glm52-model-manifest.json) 的字节摘要。
清单来自公开 HF revision API，列出全部 282 个 shard 和配置、tokenizer 文件。
创建容器前逐文件计算 hash：LFS 文件使用 SHA256，其余使用 Git blob SHA1。
`.hf-revision` 标记也必须与调用者指定 revision 一致。文件变化、缺失、清单不一致
或资源分配丢失均停止准备。每次执行需读取约 438 GB；旧验证报告不能证明共享文件
当前仍与清单一致。

固定 AIPerf 检出中的公开 Weka loader 没有 CLI revision 参数。在安装了 `datasets`
的环境中，用 [`utils/prepare_dataset.py`](../utils/prepare_dataset.py) 导出 revision
`23f152f6f0f9399a85901b89a6458def0ef16729` 的 `train` split：

```bash
python utils/prepare_dataset.py \
  --repo semianalysisai/cc-traces-weka-062126 \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 \
  --output /share_nfs/gguasti/inferencex/datasets/cc-traces-weka-062126-23f152f6f0f9399a85901b89a6458def0ef16729
```

目标目录必须不存在。导出用编号 JSON 文件保留原始行顺序，生成 `manifest.json`
和 SHA256。在请求或手动触发输入中提供 `dataset-manifest-sha256`。
检查程序验证全部文件字节并拒绝额外 trace JSON，以只读方式挂载到 `/dataset`。
配方选择 AIPerf 原生 `--input-file /dataset/traces --custom-dataset-type weka_trace`
路径；此资源分配路径不再下载可变的公开数据集。仍使用现有 Weka 重建、上下文过滤
和 trace 选择逻辑；实际 CI 中的端到端消费尚待验证。
Replay tokenizer 也从已验证的 `/model` 挂载读取，不再独立解析 HF 仓库的当前版本。

`allocation-model-validation.json` 和 `allocation-dataset-validation.json` 与资源分配
证据一起上传。2026-10-14 的准备已验证全部 288 个模型文件和 393 个导出 trace。
这些是文件完整性检查，不是 serving 或性能测量。

Runner 生命周期监督器必须在资源分配结束时停止监听进程并移除注册，包括从未接收
作业的空闲 runner。容器检查不能替代该监督器。宣布集成完成前，必须验证资源分配
丢失、runner 终止、产物上传以及镜像的非 root 运行兼容性。
