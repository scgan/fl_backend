# 原文分块策略

单篇和批量索引共同调用 qdrant_utils._blocks_to_chunks。
默认目标 512 token、硬上限 768 token、重叠约 64 token，短尾块阈值 128 token。
预算包含 text-embedding-v4 的每条输入固定 1 token 开销。
采用随镜像发布的 Qwen/Qwen3-Embedding-0.6B tokenizer，不使用英语 tokenizer 或字符估算。
本地词表对中、英、阿拉伯、日、俄、泰、印地语和特殊字符的计数已与接口 usage 校准；
每批向量化继续核对实际用量，出现差异立即报错，防止模型更新后悄悄使用错误预算。

优先在目标附近选择各语言句尾或空行；没有可用句尾时按段落。
DOCX/TXT 的解析 block 是段落，PDF 行 block 不视为段落。
超长段落才按 token 对应的 Unicode 字符偏移硬切，切口重新编码以核验上限。
重叠优先选完整句子的起点（约 32–96 token）；无合适边界时重叠约 64 token。
尾块在上限允许时合并；不为凑最小长度破坏已经选好的自然边界。

字符偏移仍对应拼接后的提取原文，重叠区偏移也重叠。
每块携带与字符区间相交的原始行 bbox，PDF 页偏移只应用一次。
不推算字符级坐标。当前解析器不提供可靠章节结构，本策略不保证章节隔离，
也不能修复双栏读取顺序或 OCR 误识别。每个附件独立分块，不跨文献或附件。

Qdrant 增加 token_count、tokenizer、chunking_strategy 元数据，原定位字段兼容。
Qdrant 和 OpenSearch 从同一组分块重建，使用同一 build_id 和 content_hash。
512/768/64 是统一基线，检索效果需用各语言的真实问题进一步评估。

验证：python -m unittest discover -s tests -p test_*.py。
测试库重建工具：scripts/rechunk_test_knowledge_base.py，支持 plan/enqueue/status/verify/repair，
repair 直接持锁构建，并撤销目标文献旧队列版本，避免旧消费者使用旧策略。
要求显式 base-id 和 report；写入仅允许 foreign_language_test/corpus_chunks/foreign_chunks_v1。

本地 compose 为 Backend/Worker 设置 `FOREIGN_INDEX_QUEUE_STATUS=PENDING_TOKENS`。
排队状态对页面仍显示 PENDING；旧版消费者只领取 PENDING，无法抢走本地新请求。
生产环境默认仍为 PENDING。重试和超时回收沿用本地配置的队列状态。
