# 全局 AI 检索 SQL 与结果快照缓存

`POST /ai_search/nl-search` 的首次检索使用 `question` 生成 SQL，执行成功后将 SQL
写入现有 Redis 客户端，并在 `data.search_id` 返回服务端生成的随机标识。
缓存键为 `ai_search:sql:{当前用户ID}:{search_id}`，值仅为 SQL 字符串，固定十五分钟过期。

同一首次检索还会把去重后的完整结果写成快照，键为
`ai_search:result:{当前用户ID}:{search_id}`，值为结果列表的 JSON，同样固定十五分钟过期。
只有结果条数不超过 `AI_SEARCH_RESULT_LIMIT`（默认 800）时才写快照，避免大结果集长期占用
Redis 内存；超过上限时只保留 SQL，翻页仍按 SQL 回源查询。

首次请求：

```json
{"question":"请帮我搜索双减相关的论文","page_num":1,"page_size":10}
```

后续翻页、调整页容量或全选时，沿用首次响应中的 `search_id`：

```json
{"search_id":"首次响应中的32位标识","page_num":2,"page_size":10}
```

携带 `search_id` 时不再读取问题生成 SQL。翻页优先读取结果快照：命中后既不查库也不重复
去重，只在快照上做 Python 分页，因此翻页与全选的耗时基本恒定。快照缺失（首次即超过上限、
被淘汰或写入失败）时会用缓存的 SQL 回源查询一次，并按原规则重新写入快照。
显式重新点击搜索时不携带旧标识，即使问题相同也会创建新的检索。
前后端需要一起更新，旧客户端不传标识时仍按首次搜索处理。

缓存不存在、过期或不属于当前用户时返回 HTTP 410，提示重新搜索，不自动调用模型。
Redis 读取或写入失败时返回 HTTP 503；SQL 执行失败不会写入任何缓存。
Redis 写入失败也不会返回没有可用缓存的成功结果。

离线验证：在 Backend 的依赖环境中执行
`python -m unittest discover -s tests -p test_ai_search_cache.py -v`；
在 Frontend 执行 `node --test tests/file-search-selection.test.cjs`。
