export type Locale = "zh" | "en";

export const DEFAULT_LOCALE: Locale = "zh";
export const LOCALE_STORAGE_KEY = "diet_expert_locale";

export function normalizeLocale(value: string | null | undefined): Locale {
  return value?.trim().toLowerCase() === "en" ? "en" : "zh";
}

const zh = {
  title: "中西医饮食健康助手",
  newChat: "新对话",
  newChatTitle: "开始一个新的 session_id，当前会话的历史仍保留在数据库里",
  currentUser: "当前用户：",
  newUserNamePlaceholder: "新用户名字",
  addUser: "+ 添加用户",
  creatingUser: "创建中…",
  loadingHistory: "加载历史消息中…",
  historyFailed: "历史消息加载失败，仍可继续聊天",
  placeholder: "今天该吃什么？",
  send: "发送",
  languageLabel: "语言",
  clarificationBanner: "需要补充一点信息",
  taskLabel: "子任务 {index}/{total}",
  stageActiveSuffix: "…",
  archivedSummaryPrefix: "（历史摘要，原文已归档）",
  confirmPending: "确认写入画像",
  revokePending: "忽略",
  defaultPendingDetail: "检测到一条关键事实，确认后才会写入画像。",
  httpError: "请求失败（HTTP {status}）",
  networkErrorNoBody: "响应没有 body（后端没有走流式响应？）",
  confirmFailed: "确认失败，请重试",
  revokeFailed: "忽略失败，请重试",
  createUserFailed: "创建用户失败，请重试",
  networkError: "网络错误，请重试",
  langZh: "中",
  langEn: "EN",
};

const en: typeof zh = {
  title: "Integrative Diet & Nutrition Advisor",
  newChat: "New chat",
  newChatTitle: "Start a new session_id. History for this conversation stays in the database.",
  currentUser: "Current user:",
  newUserNamePlaceholder: "New user name",
  addUser: "+ Add user",
  creatingUser: "Creating…",
  loadingHistory: "Loading history…",
  historyFailed: "Could not load history; you can still chat",
  placeholder: "What should I eat today?",
  send: "Send",
  languageLabel: "Language",
  clarificationBanner: "A bit more information is needed",
  taskLabel: "Subtask {index}/{total}",
  stageActiveSuffix: "…",
  archivedSummaryPrefix: "(archived summary; original text was folded)",
  confirmPending: "Confirm to profile",
  revokePending: "Ignore",
  defaultPendingDetail: "A critical fact was detected. Confirm before it is written to your profile.",
  httpError: "Request failed (HTTP {status})",
  networkErrorNoBody: "Response had no body (backend did not stream?)",
  confirmFailed: "Confirm failed; please retry",
  revokeFailed: "Ignore failed; please retry",
  createUserFailed: "Could not create the user; please retry",
  networkError: "Network error; please retry",
  langZh: "中",
  langEn: "EN",
};

export const messages: Record<Locale, typeof zh> = { zh, en };
