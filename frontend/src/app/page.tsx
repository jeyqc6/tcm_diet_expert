"use client";

/**
 * 前端最小闭环（roadmap 阶段4.2任务12）：一个聊天页面，调用后端
 * `POST /api/chat`（SSE 流式响应，api/main.py），渲染 token/source/guardrail/done
 * 四类事件——对应 docs/ARCHITECTURE.md §10.3 的事件设计、§7 PRD 的"溯源可展开"。
 *
 * fetch() 原生不解析 SSE（浏览器的 EventSource 只支持 GET，这里是 POST 带
 * JSON body，用不了 EventSource），这里手动读 ReadableStream、按
 * "event:...\ndata:...\n\n" 的格式自己切块解析——和后端 api/main.py 里
 * `_sse_event()` 拼的格式一一对应，两边改格式要同步改。
 *
 * The backend still streams source events for tracing, but source IDs are intentionally
 * kept out of the user-facing chat bubbles because the original chunk endpoint is not
 * available in the current route list.
 *
 * 历史消息：`session_id` 只在 `POST /api/chat` 请求体里用来给后端的压缩/
 * 归档算法(§4.4.1)开记账窗口，不代表"一段独立的对话"——同一个用户点"新对话"
 * 之后仍然是同一个人、同一份画像和历史，没道理刷新页面/切会话之后就看不到
 * 之前说过的话。所以挂载时拉的是不按 session_id 过滤的 `GET /api/messages`
 * （`load_all_messages()`），返回这个用户名下所有 session 的全部轮次，按
 * 时间顺序拼成一条完整的聊天记录。已归档(Tier2/Tier3)的轮次原始用户提问
 * 已经不在库里了（D27 归档设计如此），对应行 `user_text` 为 `null`——这类
 * 轮次不伪造用户气泡，只展示一条"历史摘要"提示。
 *
 * Markdown 渲染（2026-08-30 新增）：SubAgent/调和层的 prompt 里一直允许模型
 * 用 `**加粗**`/`- 列表`/`### 标题` 这类 markdown 语法组织输出（比如
 * `backend/skills/recipe_and_shopping_list.md` 的菜谱/购物清单模板），但这
 * 之前完全没有渲染——`turn.text` 原样当纯文本塞进 `<div>`，用户看到的是带
 * `#`/`*` 符号的原始 markdown 源码，不是格式化后的效果。`renderMarkdown()`
 * 是一个只处理"我们的 prompt 实际会产出的那几种语法"的最小渲染器（标题/
 * 加粗/无序列表/段落），不是通用 markdown 解析器——没有引入 react-markdown
 * 这类依赖，因为需要覆盖的语法子集很小，够用就不必换一整套解析生态。
 *
 * 用户切换（2026-08-30 新增）：`user_id` 才是真正的隔离键——`session_id`
 * 只是同一个用户名下的压缩记账单位，`GET/POST /api/users` 才是"添加一个人/
 * 列出已有的人"。`localStorage` 存两样东西：当前选中的 `user_id`，以及
 * 每个 `user_id` 各自的 `session_id`（切换用户不应该复用别人的 session_id——
 * 那会导致这个 session 底下混进两个人的对话，`conversation_sessions.user_id`
 * 只认第一次 `record_turn()` 时写进去的那个值）。
 */
import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import {
  DEFAULT_LOCALE,
  LOCALE_STORAGE_KEY,
  messages,
  normalizeLocale,
  type Locale,
} from "./messages";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8123";
const CURRENT_USER_STORAGE_KEY = "diet_expert_current_user_id";

function sessionStorageKeyFor(userId: string): string {
  return `diet_expert_session_id::${userId}`;
}

type UserInfo = { user_id: string; name: string };

type Guardrail = { type: string; detail?: string; reason?: string };

type PendingFact = {
  pending_id: string;
  allergens: string[];
  supplements: string[];
  detail: string;
};

type ChatTurn = {
  role: "user" | "assistant";
  text: string;
  sources: string[];
  guardrails: Guardrail[];
  done: boolean;
  archivedSummary?: boolean; // 历史摘要提示气泡（Tier2/3，原文已不在库里），不是真实一轮问答
  clarification?: boolean;
  taskLabel?: string;
};

type SessionMessage = {
  turn_index: number;
  archived: boolean;
  branch: string;
  user_text: string | null;
  assistant_text: string;
  cited_source_ids: string[];
  triggered_guardrails: string[];
};

function parseSseBlock(block: string): { event: string; data: string } | null {
  const eventMatch = block.match(/^event:\s*(.+)$/m);
  const dataMatch = block.match(/^data:\s*(.+)$/m);
  if (!eventMatch || !dataMatch) return null;
  return { event: eventMatch[1].trim(), data: dataMatch[1].trim() };
}

const CITATION_MARKER_PATTERN =
  /[ \t]*\[source:\s*[A-Za-z0-9_-]+(?:\s*[,，]\s*[A-Za-z0-9_-]+)*\]/gi;

function stripCitationMarkers(text: string): string {
  // Keep machine-readable citations out of both live and historical chat text.
  return text
    .replace(CITATION_MARKER_PATTERN, "")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/[ \t]+([，。！？；：、,.!?;:])/g, "$1")
    .trim();
}

// 只处理这几种语法：**加粗**——SubAgent/调和层的 prompt 里没有要求斜体/
// 行内代码，不支持的语法原样保留星号，不会报错也不会吞掉内容。
function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") && part.length > 4 ? (
      <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      <Fragment key={`${keyPrefix}-${i}`}>{part}</Fragment>
    ),
  );
}

// 最小 markdown 渲染器，只覆盖 backend/skills/*.md 实际会让模型产出的语法：
// `#`-`######` 标题、`-`/`*` 无序列表、`**加粗**`、以及普通段落(空行分段，
// 段内单个换行渲染成 <br/>)。不是通用 markdown 解析器，故意的——见本文件
// 顶部模块文档"Markdown 渲染"一节。
function renderMarkdown(text: string): ReactNode[] {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    const headerMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      // Capture before JSX: the automatic runtime evaluates children
      // (which used to `key++`) before the `key` prop, so a heading
      // followed by a paragraph reused the same React key.
      const blockKey = key++;
      blocks.push(
        <div
          key={blockKey}
          style={{ fontWeight: 700, fontSize: level <= 2 ? 15 : 13, margin: "10px 0 4px" }}
        >
          {renderInlineMarkdown(headerMatch[2], `h${blockKey}`)}
        </div>,
      );
      i++;
      continue;
    }

    const listMatch = line.match(/^[-*]\s+(.*)$/);
    if (listMatch) {
      const items: string[] = [];
      while (i < lines.length) {
        const m = lines[i].match(/^[-*]\s+(.*)$/);
        if (!m) break;
        items.push(m[1]);
        i++;
      }
      const blockKey = key++;
      blocks.push(
        <ul key={blockKey} style={{ margin: "4px 0", paddingLeft: 20 }}>
          {items.map((item, idx) => (
            <li key={idx}>{renderInlineMarkdown(item, `l${blockKey}-${idx}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    const paraLines: string[] = [];
    while (i < lines.length && lines[i].trim() !== "" && !lines[i].match(/^(#{1,6})\s+/) && !lines[i].match(/^[-*]\s+/)) {
      paraLines.push(lines[i]);
      i++;
    }
    const blockKey = key++;
    blocks.push(
      <p key={blockKey} style={{ margin: "4px 0" }}>
        {paraLines.map((l, idx) => (
          <Fragment key={idx}>
            {renderInlineMarkdown(l, `p${blockKey}-${idx}`)}
            {idx < paraLines.length - 1 && <br />}
          </Fragment>
        ))}
      </p>,
    );
  }

  return blocks;
}

function loadOrCreateSessionId(userId: string): string {
  // SSR 阶段没有 localStorage；这个函数只在 `useState` 初始化器/`useEffect`
  // 里调用，两者都只在浏览器端跑，不需要额外的 `typeof window` 判断。
  const key = sessionStorageKeyFor(userId);
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const created = crypto.randomUUID();
  window.localStorage.setItem(key, created);
  return created;
}

function messagesToTurns(rows: SessionMessage[], archivedPrefix: string): ChatTurn[] {
  const turns: ChatTurn[] = [];
  for (const m of rows) {
    if (m.archived) {
      turns.push({
        role: "assistant",
        text: `${archivedPrefix}${stripCitationMarkers(m.assistant_text)}`,
        sources: m.cited_source_ids,
        guardrails: m.triggered_guardrails.map((type) => ({ type })),
        done: true,
        archivedSummary: true,
      });
      continue;
    }
    if (m.user_text) {
      turns.push({ role: "user", text: m.user_text, sources: [], guardrails: [], done: true });
    }
    turns.push({
      role: "assistant",
      text: stripCitationMarkers(m.assistant_text),
      sources: m.cited_source_ids,
      guardrails: m.triggered_guardrails.map((type) => ({ type })),
      done: true,
    });
  }
  return turns;
}

export default function Home() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [pendingFacts, setPendingFacts] = useState<PendingFact[]>([]);
  const [pendingBusy, setPendingBusy] = useState<string | null>(null);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [currentUserId, setCurrentUserId] = useState<string>("");
  const [newUserName, setNewUserName] = useState("");
  const [creatingUser, setCreatingUser] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  // Keep the first render identical on the server and client; localStorage is
  // applied after hydration so a saved locale does not cause a mismatch.
  const [locale, setLocale] = useState<Locale>(DEFAULT_LOCALE);
  const sessionId = useRef<string>("");
  const userId = useRef<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const copy = messages[locale];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: sending ? "auto" : "smooth",
      block: "end",
    });
  }, [turns, pendingFacts, sending]);

  // 加载"这个用户全部历史"——挂载时、以及切换/新建用户之后都要调这个，
  // 抽成一个函数避免两处各写一遍拼接逻辑。不按 `ignore` 之外的条件比对
  // "是不是还是当前用户"：调用方(mount effect/switchUser)各自负责判断要不
  // 要丢弃这次请求的结果。
  async function fetchHistoryForUser(uid: string): Promise<SessionMessage[]> {
    const resp = await fetch(`${API_BASE}/api/messages?user_id=${encodeURIComponent(uid)}`);
    if (!resp.ok) throw new Error(`history request failed: ${resp.status}`);
    const body: { messages: SessionMessage[] } = await resp.json();
    return body.messages ?? [];
  }

  useEffect(() => {
    const stored = normalizeLocale(window.localStorage.getItem(LOCALE_STORAGE_KEY));
    document.documentElement.lang = stored;
    document.title = messages[stored].title;
    if (stored === DEFAULT_LOCALE) return;

    const timer = window.setTimeout(() => setLocale(stored), 0);
    return () => window.clearTimeout(timer);
  }, []);

  function switchLocale(next: Locale) {
    setLocale(next);
    window.localStorage.setItem(LOCALE_STORAGE_KEY, next);
    document.documentElement.lang = next;
    document.title = messages[next].title;
  }

  useEffect(() => {
    // React(dev 模式下的 Strict Mode)会把这个 effect mount→cleanup→再mount
    // 一遍来暴露没写 cleanup 的副作用——如果不设这个 `ignore` 标记，两次
    // effect 各自发起的请求都会成功、各自 setTurns 一次，历史消息会在页面上
    // 重复出现两份。这是 React 官方文档"在 Effect 里请求数据"给出的标准写法，
    // 不是这个项目专门发明的模式。
    let ignore = false;
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/users`);
        if (!resp.ok) throw new Error(`users request failed: ${resp.status}`);
        if (ignore) return; // A stale effect must not update the current screen.
        const body: { users: UserInfo[] } = await resp.json();
        const fetchedUsers = body.users ?? [];
        if (ignore) return;
        setUsers(fetchedUsers);
        if (fetchedUsers.length === 0) return; // 极端情况(全新空库)：先不选任何用户，等用户自己创建

        const stored = window.localStorage.getItem(CURRENT_USER_STORAGE_KEY);
        const initialUserId =
          stored && fetchedUsers.some((u) => u.user_id === stored) ? stored : fetchedUsers[0].user_id;
        userId.current = initialUserId;
        setCurrentUserId(initialUserId);
        window.localStorage.setItem(CURRENT_USER_STORAGE_KEY, initialUserId);
        sessionId.current = loadOrCreateSessionId(initialUserId);

        const rows = await fetchHistoryForUser(initialUserId);
        // 用户可能在这次请求还没返回时就已经发了第一条消息(乐观更新已经把
        // user/assistant 气泡塞进 `turns`)——这里改成"往前拼接"而不是整体
        // 覆盖，避免拿一份历史响应把用户刚发的新消息气泡冲掉。
        if (ignore) return;
        const activeLocale = normalizeLocale(window.localStorage.getItem(LOCALE_STORAGE_KEY));
        setTurns((prev) => [...messagesToTurns(rows, messages[activeLocale].archivedSummaryPrefix), ...prev]);
      } catch {
        // 同上：网络错误不阻塞——用户仍然可以正常发新消息
        if (!ignore) {
          const activeLocale = normalizeLocale(window.localStorage.getItem(LOCALE_STORAGE_KEY));
          setErrorMessage(messages[activeLocale].historyFailed);
        }
      } finally {
        if (!ignore) setHistoryLoaded(true);
      }
    })();
    return () => {
      ignore = true;
    };
  }, []);

  async function switchUser(uid: string) {
    if (uid === userId.current || sending) return;
    userId.current = uid;
    setCurrentUserId(uid);
    window.localStorage.setItem(CURRENT_USER_STORAGE_KEY, uid);
    sessionId.current = loadOrCreateSessionId(uid);
    setTurns([]);
    setPendingFacts([]);
    setHistoryLoaded(false);
    try {
      const rows = await fetchHistoryForUser(uid);
      if (userId.current !== uid) return; // 拉取过程中又切换了一次，这份结果已经过期
      setTurns(messagesToTurns(rows, copy.archivedSummaryPrefix));
    } catch {
      // 网络错误不阻塞——用户仍然可以正常发新消息，只是历史没加载出来
      if (userId.current === uid) setErrorMessage(copy.historyFailed);
    } finally {
      if (userId.current === uid) setHistoryLoaded(true);
    }
  }

  async function createUser() {
    const name = newUserName.trim();
    if (!name || creatingUser) return;
    setCreatingUser(true);
    try {
      const resp = await fetch(`${API_BASE}/api/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!resp.ok) {
        setErrorMessage(copy.createUserFailed);
        return;
      }
      const created: UserInfo = await resp.json();
      setUsers((prev) => [...prev, created]);
      setNewUserName("");
      await switchUser(created.user_id);
    } catch {
      // 网络错误：新用户没建成，输入框内容保留，用户可以重试
    } finally {
      setCreatingUser(false);
    }
  }

  function startNewConversation() {
    const created = crypto.randomUUID();
    window.localStorage.setItem(sessionStorageKeyFor(userId.current), created);
    sessionId.current = created;
    setTurns([]);
    setPendingFacts([]);
  }

  async function sendMessage() {
    const message = input.trim();
    if (!message || sending) return;
    setInput("");
    setSending(true);

    setTurns((prev) => [
      ...prev,
      { role: "user", text: message, sources: [], guardrails: [], done: true },
      { role: "assistant", text: "", sources: [], guardrails: [], done: false },
    ]);

    try {
      const resp = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId.current,
          message,
          user_id: userId.current,
          locale,
        }),
      });
      if (!resp.ok) {
        applyEvent(
          "guardrail",
          JSON.stringify({
            type: "http_error",
            detail: copy.httpError.replace("{status}", String(resp.status)),
          }),
        );
        applyEvent("done", JSON.stringify({}));
        return;
      }
      if (!resp.body) throw new Error(copy.networkErrorNoBody);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? ""; // 最后一块可能还不完整，留到下一轮再拼

        for (const block of blocks) {
          const parsed = parseSseBlock(block);
          if (!parsed) continue;
          applyEvent(parsed.event, parsed.data);
        }
      }
    } catch {
      applyEvent("guardrail", JSON.stringify({ type: "network_error", detail: copy.networkError }));
      applyEvent("done", JSON.stringify({}));
    } finally {
      setSending(false);
    }
  }

  function applyEvent(event: string, dataText: string) {
    let data: Record<string, unknown> = {};
    try {
      data = JSON.parse(dataText);
    } catch {
      // 解析失败就当成纯文本处理，不让一条格式异常的事件炸掉整个渲染
    }

    if (event === "critical_fact_pending") {
      const pendingId = String(data.pending_id ?? "");
      if (pendingId) {
        setPendingFacts((prevFacts) => {
          if (prevFacts.some((p) => p.pending_id === pendingId)) return prevFacts;
          return [
            ...prevFacts,
            {
              pending_id: pendingId,
              allergens: Array.isArray(data.allergens) ? data.allergens.map(String) : [],
              supplements: Array.isArray(data.supplements) ? data.supplements.map(String) : [],
                  detail: String(data.detail ?? copy.defaultPendingDetail),
            },
          ];
        });
      }
      return;
    }

    setTurns((prev) => {
      const last = prev[prev.length - 1];
      if (!last || last.role !== "assistant") return prev;

      // 必须返回全新对象，不能就地改 `last`——React 18 Strict Mode 开发环境下
      // setState 的 updater 函数会被调用两次（用来暴露非纯的 updater），如果这里
      // 改的是 `prev` 里那个共享的对象引用，第二次调用会在已经改过一次的基础上
      // 再改一次，token/source/guardrail 全部重复一份——这正是页面上看到的
      // 文字整段错位重叠的原因。
      let updated: ChatTurn = last;
      if (event === "token") {
        updated = { ...last, text: last.text + String(data.text ?? "") };
      } else if (event === "source") {
        const sourceId = String(data.source_id ?? "");
        if (sourceId && !last.sources.includes(sourceId)) {
          updated = { ...last, sources: [...last.sources, sourceId] };
        }
      } else if (event === "guardrail") {
        updated = {
          ...last,
          guardrails: [
            ...last.guardrails,
            {
              type: String(data.type ?? "unknown"),
              detail: data.detail ? String(data.detail) : undefined,
              reason: data.reason ? String(data.reason) : undefined,
            },
          ],
        };
      } else if (event === "clarification") {
        updated = {
          ...last,
          clarification: true,
        };
      } else if (event === "task") {
        const index = Number(data.index ?? 0) + 1;
        const total = Number(data.total ?? 1);
        const branch = String(data.branch ?? "");
        updated = {
          ...last,
          taskLabel: `${copy.taskLabel.replace("{index}", String(index)).replace("{total}", String(total))}${branch ? ` · ${branch}` : ""}`,
        };
      } else if (event === "task_done") {
        updated = { ...last, done: true };
      } else if (event === "done") {
        updated = { ...last, done: true };
      }
      if (updated === last) return prev;
      return [...prev.slice(0, -1), updated];
    });
  }

  async function decidePendingFact(pendingId: string, action: "confirm" | "revoke") {
    if (pendingBusy) return;
    setPendingBusy(pendingId);
    try {
      const resp = await fetch(`${API_BASE}/api/profile/critical-facts/${action}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pending_id: pendingId }),
      });
      if (!resp.ok) {
        applyEvent(
          "guardrail",
          JSON.stringify({
            type: "critical_fact_error",
            detail: action === "confirm" ? copy.confirmFailed : copy.revokeFailed,
          }),
        );
        return;
      }
      setPendingFacts((prev) => prev.filter((p) => p.pending_id !== pendingId));
    } catch {
      applyEvent(
        "guardrail",
        JSON.stringify({ type: "network_error", detail: copy.networkError }),
      );
    } finally {
      setPendingBusy(null);
    }
  }

  return (
    <div
      style={{
        maxWidth: 720,
        minHeight: "100vh",
        boxSizing: "border-box",
        margin: "0 auto",
        padding: "168px 24px 0",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          position: "fixed",
          top: 0,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 10,
          boxSizing: "border-box",
          width: "calc(100% - 48px)",
          maxWidth: 672,
          padding: "16px 0 12px",
          background: "white",
          borderBottom: "1px solid #eee",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <h1 style={{ fontSize: 20, margin: 0 }}>{copy.title}</h1>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
          <div
            role="group"
            aria-label={copy.languageLabel}
            style={{ display: "flex", border: "1px solid #ccc", borderRadius: 8, overflow: "hidden" }}
          >
            <button
              type="button"
              onClick={() => switchLocale("zh")}
              aria-pressed={locale === "zh"}
              style={{
                padding: "6px 10px",
                border: "none",
                background: locale === "zh" ? "#0070f3" : "white",
                color: locale === "zh" ? "white" : "#333",
                fontSize: 13,
                cursor: "pointer",
              }}
            >
              {copy.langZh}
            </button>
            <button
              type="button"
              onClick={() => switchLocale("en")}
              aria-pressed={locale === "en"}
              style={{
                padding: "6px 10px",
                border: "none",
                background: locale === "en" ? "#0070f3" : "white",
                color: locale === "en" ? "white" : "#333",
                fontSize: 13,
                cursor: "pointer",
              }}
            >
              {copy.langEn}
            </button>
          </div>
          <button
          type="button"
          onClick={startNewConversation}
          disabled={sending}
          title={copy.newChatTitle}
          style={{
            padding: "6px 12px",
            borderRadius: 8,
            border: "1px solid #ccc",
            background: "white",
            color: "#333",
            fontSize: 13,
            whiteSpace: "nowrap",
          }}
        >
          {copy.newChat}
        </button>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, color: "#666" }}>{copy.currentUser}</span>
          <select
            value={currentUserId}
            onChange={(e) => void switchUser(e.target.value)}
            disabled={sending || users.length === 0}
            style={{ padding: "4px 8px", borderRadius: 6, border: "1px solid #ccc", fontSize: 13 }}
          >
            {users.map((u) => (
              <option key={u.user_id} value={u.user_id}>
                {u.name}
              </option>
            ))}
          </select>
          <input
            value={newUserName}
            onChange={(e) => setNewUserName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void createUser();
              }
            }}
            placeholder={copy.newUserNamePlaceholder}
            disabled={creatingUser}
            style={{ padding: "4px 8px", borderRadius: 6, border: "1px solid #ccc", fontSize: 13, width: 120 }}
          />
          <button
            type="button"
            onClick={() => void createUser()}
            disabled={creatingUser || !newUserName.trim()}
            style={{
              padding: "4px 10px",
              borderRadius: 6,
              border: "1px solid #ccc",
              background: "white",
              color: "#333",
              fontSize: 13,
            }}
          >
            {creatingUser ? copy.creatingUser : copy.addUser}
          </button>
        </div>
      </div>

      <div style={{ padding: "16px 0 96px" }}>
        {!historyLoaded && (
          <p style={{ color: "#999", fontSize: 13, marginBottom: 16 }}>{copy.loadingHistory}</p>
        )}

        {errorMessage && (
          <p role="alert" style={{ color: "#a15c00", fontSize: 13, marginBottom: 16 }}>
            {errorMessage}
          </p>
        )}

        {pendingFacts.map((fact) => (
          <div
            key={fact.pending_id}
            style={{
              marginBottom: 12,
              padding: "8px 12px",
              borderRadius: 8,
              background: "#fff4e0",
              color: "#5c3d00",
              fontSize: 13,
            }}
          >
            <div>{fact.detail}</div>
            <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
              <button
                type="button"
                disabled={pendingBusy === fact.pending_id}
                onClick={() => void decidePendingFact(fact.pending_id, "confirm")}
                style={{
                  padding: "4px 10px",
                  borderRadius: 6,
                  border: "none",
                  background: "#0070f3",
                  color: "white",
                  fontSize: 13,
                }}
              >
                {copy.confirmPending}
              </button>
              <button
                type="button"
                disabled={pendingBusy === fact.pending_id}
                onClick={() => void decidePendingFact(fact.pending_id, "revoke")}
                style={{
                  padding: "4px 10px",
                  borderRadius: 6,
                  border: "1px solid #ccc",
                  background: "white",
                  color: "#333",
                  fontSize: 13,
                }}
              >
                {copy.revokePending}
              </button>
            </div>
          </div>
        ))}

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {turns.map((turn, i) => (
            <div
              key={i}
              style={{
                alignSelf: turn.role === "user" ? "flex-end" : "flex-start",
                maxWidth: "85%",
                background: turn.role === "user" ? "#0070f3" : "#f2f2f2",
                color: turn.role === "user" ? "white" : "black",
                borderRadius: 10,
                padding: "8px 12px",
                whiteSpace: "pre-wrap",
              }}
            >
              {turn.taskLabel && (
                <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>{turn.taskLabel}</div>
              )}
              {turn.clarification && (
                <div style={{ fontSize: 11, color: "#666", marginBottom: 4 }}>{copy.clarificationBanner}</div>
              )}
              <div>
                {turn.text
                  ? renderMarkdown(stripCitationMarkers(turn.text))
                  : turn.role === "assistant" && !turn.done
                    ? "…"
                    : ""}
              </div>

              {turn.guardrails.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  {turn.guardrails.map((g, gi) => (
                    <div
                      key={gi}
                      style={{
                        fontSize: 12,
                        color: "#a15c00",
                        background: "#fff4e0",
                        borderRadius: 6,
                        padding: "4px 8px",
                        marginTop: 4,
                      }}
                    >
                      ⚠ [{g.type}] {g.detail ?? g.reason ?? ""}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
        <div ref={messagesEndRef} style={{ height: 1 }} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void sendMessage();
        }}
        style={{
          position: "fixed",
          bottom: 0,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 10,
          boxSizing: "border-box",
          width: "calc(100% - 48px)",
          maxWidth: 672,
          display: "flex",
          gap: 8,
          padding: "12px 0 16px",
          background: "white",
          borderTop: "1px solid #eee",
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={copy.placeholder}
          disabled={sending}
          style={{ flex: 1, padding: "8px 12px", borderRadius: 8, border: "1px solid #ccc" }}
        />
        <button
          type="submit"
          disabled={sending}
          style={{ padding: "8px 16px", borderRadius: 8, border: "none", background: "#0070f3", color: "white" }}
        >
          {sending ? "…" : copy.send}
        </button>
      </form>
    </div>
  );
}
