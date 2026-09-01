# 中西医饮食健康助手 · Web 前端

roadmap 阶段4.2任务12。用 `create-next-app` 建出来的最小 Next.js app(App Router,
TypeScript,不带 Tailwind——单页面，没必要引入一整套样式框架)，唯一页面
`src/app/page.tsx` 是一个能问能答的聊天界面，调用后端 `POST /api/chat`
(`api/main.py`)的 SSE 流式响应。

## 跑起来

```bash
# 1. 后端(项目根目录)
export DIET_EXPERT_PG_DSN=...   # 见根目录 .env.example
uvicorn api.main:app --port 8123

# 2. 前端
cd frontend
npm install
npm run dev   # 默认 http://localhost:3000
```

默认认为后端跑在 `http://localhost:8123`；换端口/换主机用环境变量
`NEXT_PUBLIC_API_BASE_URL`(见 `.env.local.example`)。

## 做了什么，没做什么

- ✅ 发消息、SSE 逐块渲染回答(`token`)、处理来源事件、展示
  `guardrail`——见 `docs/BUILD_PLAN.md` 阶段4 #12。
- ✅ **会话历史**：`session_id` 存在 `localStorage`，挂载时调
  `GET /api/sessions/{id}/messages` 拉回；有「新对话」按钮。已归档轮次只显示
  摘要提示，不伪造用户原话。
- ✅ **SSE 协议**：`applyEvent` 处理 `token` / `source` / `guardrail` / `done`
  / `clarification` / `task` / `task_done` / `critical_fact_pending`。
  onboarding 在 `/api/chat` 里也是 `token` 事件（同一条渲染路径）。
  `sendMessage` 检查 `resp.ok`，HTTP 4xx/5xx 显示 `http_error` guardrail。
- ✅ **关键事实确认条**：收到 `critical_fact_pending` 后展示确认/忽略，分别打
  `POST /api/profile/critical-facts/confirm` 与 `.../revoke`。未确认前不会写入
  画像，也不会影响本轮已经生成的建议（D34）。
- ⏳ **来源信息不在聊天界面展示**：后端仍处理 `source` 事件，但当前没有
  可供用户打开原文的 chunk 查询端点。
- ⏳ **没有独立的首次引导 UI**：`GET/PATCH /api/profile`、`/api/onboarding/*`
  已存在；`/api/chat` 会在需要时用 token 流问引导问题，前端没有单独的问卷页。

## 已知限制：这次没有在真实浏览器里点击验证

验证方式是：对照 `page.tsx` 的事件分支与后端 `sse_event` 名称对齐；真实启动
前后端两个 dev server 后，建议你自己打开浏览器点一下确认/忽略条和多任务
`task` 标签。这个环境若没有浏览器自动化，不会假装已经点过。
