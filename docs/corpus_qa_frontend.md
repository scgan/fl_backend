# 文献问答多轮对话 - 前端对接文档

## 一、接口地址
```
POST /api/ai_search/corpus_qa
Content-Type: application/json
```

## 二、请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 用户输入的问题 |
| `corpus_ids` | array[int] | 是 | 选中的文献 ID 列表（最多 20 篇） |
| `question_type` | string | 否 | 问答类型，默认 `custom` |
| `session_id` | string | **否** | 会话 ID（用于多轮对话） |

### `question_type` 可选值
- `custom`：自定义问题（需配合 `question`）
- `summary`：文献综述
- `comparison`：对比分析（至少选 2 篇）
- `core_idea`：核心观点
- `outline`：文章大纲
- `highlights`：关键要点
- `methodology`：研究方法

---

## 三、响应（SSE 流式事件）

### 事件 1：`meta`（元数据，最先推送）
```json
{
  "question_type": "custom",
  "question": "这篇文献的主要贡献是什么？",
  "corpus_count": 2,
  "corpus_ids": [101, 102],
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### 事件 2：`chunk`（回答片段，逐段推送）
```json
{
  "content": "这篇文献的主要贡献包括..."
}
```

### 事件 3：`done`（完成，最终信息）
```json
{
  "history_id": 42,
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### 事件 4：`error`（错误）
```json
{
  "error": "文件解析超时或API调用失败"
}
```

---

## 四、前端交互流程

### 1. 首次提问（新建会话）
- 请求体**不传** `session_id`（或不传此字段）
- 从 `meta` 或 `done` 事件中取出 `session_id`，**本地保存**
- 示例：
```javascript
async function askQuestion(question, corpusIds) {
  const response = await fetch('/api/ai_search/corpus_qa', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      corpus_ids: corpusIds,
      question_type: 'custom'
      // 首次不传 session_id
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let sessionId = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split('\n');

    for (const line of lines) {
      if (line.startsWith('event:')) {
        const eventType = line.replace('event:', '').trim();
        continue;
      }
      if (line.startsWith('data:')) {
        const data = JSON.parse(line.replace('data:', '').trim());

        // 根据事件类型处理（meta / chunk / done / error）
        // ...
      }
    }
  }
}
```

### 2. 后续追问（续写会话）
- 请求体中**带上**之前保存的 `session_id`
- 后端会自动注入最近 10 轮历史，继续上下文
```javascript
body: JSON.stringify({
  question: '追问内容',
  corpus_ids: corpusIds,
  question_type: 'custom',
  session_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'  // 带上之前的 session_id
})
```

### 3. 新会话
- 发起全新话题时，**不要传** `session_id`（或传新的 UUID）
- 后端会视为新建会话，清除上下文

---

## 五、前端需要做的事情

### 核心逻辑（伪代码）
```javascript
class QAClient {
  constructor() {
    this.sessionId = null;   // 本地保存当前会话 ID
  }

  async ask(question, corpusIds, questionType = 'custom') {
    const payload = {
      question,
      corpus_ids: corpusIds,
      question_type: questionType
    };

    // 如果有会话 ID，带上（用于多轮对话）
    if (this.sessionId) {
      payload.session_id = this.sessionId;
    }

    const response = await fetch('/api/ai_search/corpus_qa', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    // 用于拼接完整回答
    let fullAnswer = '';

    // 用于 SSE 解析
    let currentEvent = null;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split('\n');

      for (const line of lines) {
        if (line.startsWith('event:')) {
          currentEvent = line.replace('event:', '').trim();
        } else if (line.startsWith('data:')) {
          const data = JSON.parse(line.replace('data:', '').trim());

          switch (currentEvent) {
            case 'meta':
              // 首次请求时保存 session_id（后端生成）
              this.sessionId = data.session_id;
              // 可在这里展示 "正在分析..." 状态
              break;

            case 'chunk':
              fullAnswer += data.content;
              // 实时更新 UI（如打字机效果）
              this.onChunk(data.content);
              break;

            case 'done':
              // 保存本次 history_id（可选，用于后续查询）
              this.historyId = data.history_id;
              // 完成回调
              this.onDone(fullAnswer, data);
              break;

            case 'error':
              // 错误处理
              this.onError(data.error);
              break;
          }
        }
      }
    }
  }

  // 回调（可自定义）
  onChunk(content) { /* 实时显示 */ }
  onDone(answer, meta) { /* 完成处理 */ }
  onError(err) { /* 错误处理 */ }
}
```

### UI 注意事项
1. **session_id 本地持久化**：存入 `localStorage` 或 Vuex/Pinia store，刷新页面后仍可继续同一会话
2. **轮次过多时的提示**：后端自动截断到最近 10 轮，若对话轮次接近上限可提示用户"开始新会话"
3. **换话题/清空上下文**：不传 `session_id` 即可新建会话，前端可提供"新会话"按钮
4. **取消请求**：前端可使用 `AbortController` 取消正在进行的请求

---

## 六、向后兼容

- **`session_id` 完全可选**：不传此字段时，后端行为与之前完全一致（单轮对话）
- **历史记录接口不变**：`GET /api/ai_search/corpus_qa/history` 仍可按用户拉��全部历史
- **前端改动最小**：只需在请求/响应的 JSON 中加一个可选字段即可

---

## 七、数据库迁移（如需要）

如果 `corpus_qa_history` 表缺少 `session_id`、`turn` 字段，需要执行迁移：

```sql
ALTER TABLE corpus_qa_history
    ADD COLUMN session_id VARCHAR(64) NULL AFTER is_deleted,
    ADD COLUMN turn INT NULL AFTER session_id,
    ADD INDEX idx_session_id (session_id);
```

---

## 八、QA

**Q: `session_id` 如何生成？**
A: 首次请求不传 `session_id` 时，后端用 `uuid.uuid4()` 自动生成。前端也可以自己生成 UUID 传过来（需保证唯一性）。

**Q: `corpus_ids` 每次都要传吗？**
A: 是的，`corpus_ids` 每次必传（后端需要上传文件到千问 API）。但可以增减文献（新增/减少），后端会正确处理。

**Q: 历史最多保留多少轮？**
A: 后端默认保留最近 10 轮（可配置 `history_limit` 参数）。更早的轮次会自动丢弃，不会塞入上下文。

**Q: 如果用户关闭页面/刷新，历史会丢失吗？**
A: 不会。历史已存入数据库，只要前端存好 `session_id`，下次带上即可续接。

**Q: 多用户并发使用会冲突吗？**
A: ���会。`session_id` 是每个用户独立的，后端按 `user_id + session_id` 查询历史，互不影响。
