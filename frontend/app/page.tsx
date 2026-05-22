"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010/api/v1";
const STORAGE_KEY = "traffic-law-v2-auth";

type Message = {
  id?: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
};

type ThreadSummary = {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  messages?: number;
};

type AuthUser = {
  id: string;
  email: string;
  created_at: string;
};

type AuthState = {
  user: AuthUser;
  access_token: string;
};

const EMPTY_ASSISTANT: Message = {
  role: "assistant",
  content: "Chào bạn. Bạn có thể bắt đầu một đoạn chat mới hoặc mở lại đoạn chat cũ để hỏi tiếp.",
};

export default function Home() {
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);

  const [threadId, setThreadId] = useState<string>("");
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [threadsBusy, setThreadsBusy] = useState(false);
  const [query, setQuery] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState("");
  const [messages, setMessages] = useState<Message[]>([EMPTY_ASSISTANT]);
  const isComposingRef = useRef(false);

  useEffect(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw) as AuthState;
      setAuth(parsed);
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    if (!auth) return;
    void bootstrapAuth(auth);
  }, [auth?.access_token]);

  async function bootstrapAuth(current: AuthState) {
    try {
      const me = await fetch(`${API}/auth/me`, {
        headers: { Authorization: `Bearer ${current.access_token}` },
      });
      if (!me.ok) throw new Error("Phiên đăng nhập đã hết hạn.");
      const data = await me.json();
      const nextAuth = { ...current, user: data.user as AuthUser };
      setAuth(nextAuth);
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextAuth));
      await loadThreads(nextAuth.user.id);
    } catch {
      logout();
    }
  }

  async function loadThreads(userId: string) {
    setThreadsBusy(true);
    try {
      const response = await fetch(`${API}/threads?user_id=${encodeURIComponent(userId)}`);
      if (!response.ok) throw new Error("Không tải được danh sách chat.");
      const data = await response.json();
      setThreads((data.threads ?? []) as ThreadSummary[]);
    } finally {
      setThreadsBusy(false);
    }
  }

  async function handleAuth() {
    if (authBusy) return;
    setAuthBusy(true);
    setAuthError("");
    try {
      const endpoint = authMode === "login" ? "login" : "register";
      const response = await fetch(`${API}/auth/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Không đăng nhập được.");
      }
      const nextAuth = {
        user: data.user as AuthUser,
        access_token: data.access_token as string,
      };
      setAuth(nextAuth);
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextAuth));
      setEmail("");
      setPassword("");
      await loadThreads(nextAuth.user.id);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Xác thực thất bại.");
    } finally {
      setAuthBusy(false);
    }
  }

  function logout() {
    setAuth(null);
    setThreads([]);
    setThreadId("");
    setMessages([EMPTY_ASSISTANT]);
    setChatError("");
    window.localStorage.removeItem(STORAGE_KEY);
  }

  async function createThread(initialTitle?: string) {
    if (!auth) throw new Error("Bạn cần đăng nhập trước.");
    const title = initialTitle?.trim() || "Đoạn chat mới";
    const response = await fetch(
      `${API}/threads?user_id=${encodeURIComponent(auth.user.id)}&title=${encodeURIComponent(title)}`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error("Không tạo được đoạn chat mới.");
    const data = await response.json();
    const created = data as ThreadSummary;
    setThreadId(created.id);
    setMessages([EMPTY_ASSISTANT]);
    await loadThreads(auth.user.id);
    return created.id;
  }

  async function openThread(id: string) {
    setChatError("");
    const response = await fetch(`${API}/threads/${id}/messages`);
    if (!response.ok) {
      setChatError("Không tải được nội dung đoạn chat này.");
      return;
    }
    const data = await response.json();
    const threadMessages = (data.messages ?? []) as Message[];
    setThreadId(id);
    setMessages(threadMessages.length ? threadMessages : [EMPTY_ASSISTANT]);
  }

  async function send() {
    if (isSending || !auth) return;
    const text = query.trim();
    if (!text) return;
    setIsSending(true);
    setChatError("");
    setQuery("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    try {
      let id = threadId;
      if (!id) {
        id = await createThread(text.slice(0, 48));
      }
      const response = await fetch(`${API}/threads/${id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: auth.user.id, query: text, index_dir: "data/index", top_k: 5 }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Không gửi được câu hỏi.");
      }
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer ?? "Không nhận được phản hồi." }]);
      await loadThreads(auth.user.id);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Gửi tin nhắn thất bại.";
      setChatError(message);
      setMessages((prev) => [...prev, { role: "assistant", content: `Mình đang gặp lỗi: ${message}` }]);
    } finally {
      setIsSending(false);
    }
  }

  if (!auth) {
    return (
      <main className="grid min-h-screen place-items-center p-8">
        <section className="w-full max-w-[520px] rounded-[28px] border border-line bg-panel p-8 shadow-panel backdrop-blur-[14px]">
          <div className="mb-3 text-[12px] uppercase tracking-[0.16em] text-accent">Traffic Law Assistant</div>
          <h1 className="m-0 text-[40px] leading-[0.95]">Đăng nhập để bắt đầu chat</h1>
          <p className="mb-6 mt-4 leading-7 text-muted">
            Bạn có thể tạo tài khoản mới, mở lại các đoạn chat cũ và tiếp tục hỏi trong cùng mạch hội thoại.
          </p>

          <div className="mb-[22px] inline-flex rounded-full bg-[rgba(21,35,29,0.06)] p-1.5">
            <button
              className={
                authMode === "login"
                  ? "cursor-pointer rounded-full bg-white px-4 py-2.5 text-ink shadow-[0_8px_24px_rgba(21,35,29,0.08)]"
                  : "cursor-pointer rounded-full bg-transparent px-4 py-2.5 text-muted"
              }
              onClick={() => setAuthMode("login")}
            >
              Đăng nhập
            </button>
            <button
              className={
                authMode === "register"
                  ? "cursor-pointer rounded-full bg-white px-4 py-2.5 text-ink shadow-[0_8px_24px_rgba(21,35,29,0.08)]"
                  : "cursor-pointer rounded-full bg-transparent px-4 py-2.5 text-muted"
              }
              onClick={() => setAuthMode("register")}
            >
              Đăng ký
            </button>
          </div>

          <label className="mb-4 grid gap-2">
            <span className="text-sm text-muted">Email</span>
            <input
              className="w-full rounded-[18px] border border-line bg-field px-[18px] py-4 outline-none transition focus:border-[rgba(15,118,110,0.35)] focus:shadow-[0_0_0_3px_rgba(15,118,110,0.08)]"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
            />
          </label>

          <label className="mb-4 grid gap-2">
            <span className="text-sm text-muted">Mật khẩu</span>
            <input
              className="w-full rounded-[18px] border border-line bg-field px-[18px] py-4 outline-none transition focus:border-[rgba(15,118,110,0.35)] focus:shadow-[0_0_0_3px_rgba(15,118,110,0.08)]"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Tối thiểu 6 ký tự"
            />
          </label>

          {authError ? <p className="mt-2 text-sm text-[#9a3412]">{authError}</p> : null}

          <button
            className="mt-5 w-full cursor-pointer rounded-full bg-accent-primary px-[22px] py-3.5 text-white shadow-[0_16px_32px_rgba(15,118,110,0.18)] transition duration-150 hover:-translate-y-px disabled:cursor-not-allowed disabled:opacity-70"
            onClick={handleAuth}
            disabled={authBusy}
          >
            {authBusy ? "Đang xử lý..." : authMode === "login" ? "Đăng nhập" : "Tạo tài khoản"}
          </button>
        </section>
      </main>
    );
  }

  return (
    <main className="grid h-screen grid-cols-1 border-t-[10px] border-accent-strong md:grid-cols-[340px_minmax(0,1fr)]">
      <aside className="app-scrollbar overflow-visible border-b border-line bg-panel p-5 backdrop-blur-[14px] md:overflow-y-auto md:border-b-0 md:border-r md:p-7">
        <div className="mb-5 inline-flex w-full flex-col items-start gap-[3px] rounded-[18px] border border-[rgba(15,118,110,0.18)] bg-[rgba(255,255,255,0.62)] px-4 py-3 shadow-[0_12px_28px_rgba(21,35,29,0.06)]">
          <span className="text-xs uppercase tracking-[0.08em] text-accent">Xin chào,</span>
          <strong className="max-w-full break-words text-left text-sm">{auth.user.email}</strong>
        </div>

        <div className="flex items-start justify-start">
          <button
            className="w-full cursor-pointer rounded-full border border-line bg-[rgba(255,255,255,0.7)] px-[22px] py-3.5 text-ink transition duration-150 hover:-translate-y-px"
            onClick={logout}
          >
            Đăng xuất
          </button>
        </div>

        <div className="my-5 h-px w-full bg-gradient-to-r from-transparent via-[rgba(15,118,110,0.28)] to-transparent" />

        <button
          className="w-full cursor-pointer rounded-full bg-accent-primary px-[22px] py-3.5 text-white shadow-[0_16px_32px_rgba(15,118,110,0.18)] transition duration-150 hover:-translate-y-px"
          onClick={() => void createThread()}
        >
          Đoạn chat mới
        </button>

        <div className="mt-4 grid gap-3.5">
          <div className="flex items-center gap-3 text-sm text-muted">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-[rgba(15,118,110,0.22)] to-[rgba(15,118,110,0.08)]" />
            <span className="shrink-0 text-sm">Lịch sử chat</span>
            <div className="h-px flex-1 bg-gradient-to-r from-[rgba(15,118,110,0.08)] via-[rgba(15,118,110,0.22)] to-transparent" />
          </div>
          <div className="grid gap-2.5">
            {threads.map((thread) => (
              <button
                key={thread.id}
                className={
                  thread.id === threadId
                    ? "cursor-pointer rounded-[20px] border border-[rgba(15,118,110,0.35)] bg-[linear-gradient(135deg,rgba(15,118,110,0.08),rgba(255,255,255,0.92))] px-4 py-[14px] text-left shadow-[0_10px_24px_rgba(21,35,29,0.05)] transition duration-150 hover:-translate-y-px"
                    : "cursor-pointer rounded-[20px] border border-line bg-[rgba(255,255,255,0.72)] px-4 py-[14px] text-left shadow-[0_10px_24px_rgba(21,35,29,0.05)] transition duration-150 hover:-translate-y-px"
                }
                onClick={() => void openThread(thread.id)}
              >
                <strong className="mb-1.5 block">{thread.title || "Đoạn chat mới"}</strong>
                <span className="block text-[13px] text-muted">{new Date(thread.created_at).toLocaleString("vi-VN")}</span>
              </button>
            ))}
            {!threads.length && !threadsBusy ? <p className="text-sm text-muted">Chưa có đoạn chat nào.</p> : null}
          </div>
        </div>
      </aside>

      <section className="flex h-auto w-full min-h-0 flex-col overflow-visible p-5 md:h-screen md:overflow-hidden md:p-8">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div className="max-w-[220px] text-[26px] font-bold leading-none">Traffic Law Assistant</div>
        </div>

        <div className="mb-5 h-px w-full bg-gradient-to-r from-[rgba(15,118,110,0.22)] via-[rgba(21,35,29,0.1)] to-transparent" />

        <div className="app-scrollbar flex min-h-0 flex-1 flex-col gap-4 overflow-visible pr-0 md:overflow-y-auto md:pr-2">
          {messages.map((message, index) => (
            <div
              className={message.role === "user" ? "flex w-full justify-end" : "flex w-full justify-start"}
              key={message.id ?? `${message.role}-${index}`}
            >
              <div
                className={
                  message.role === "user"
                    ? "w-fit max-w-[78%] min-w-0 rounded-[24px] border border-line bg-accent-primary px-5 py-[18px] text-white shadow-panel md:max-w-[70%]"
                    : "w-fit max-w-[78%] min-w-0 rounded-[24px] border border-line bg-[rgba(255,250,240,0.9)] px-5 py-[18px] shadow-panel md:max-w-[70%]"
                }
              >
                <FormattedMessage content={message.content} />
              </div>
            </div>
          ))}
        </div>

        {chatError ? <p className="mt-2 text-sm text-[#9a3412]">{chatError}</p> : null}

        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
          <input
            className="flex-1 rounded-full border border-line bg-field px-[18px] py-4 outline-none transition focus:border-[rgba(15,118,110,0.35)] focus:shadow-[0_0_0_3px_rgba(15,118,110,0.08)]"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onCompositionStart={() => {
              isComposingRef.current = true;
            }}
            onCompositionEnd={() => {
              isComposingRef.current = false;
            }}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              if (isComposingRef.current || event.nativeEvent.isComposing) return;
              event.preventDefault();
              void send();
            }}
            placeholder="Ví dụ: Tôi muốn hỏi về mức phạt đi ngược chiều bằng xe máy"
          />
          <button
            className="cursor-pointer rounded-full bg-accent px-[22px] py-4 text-white transition duration-150 hover:-translate-y-px disabled:cursor-not-allowed disabled:opacity-70 sm:w-auto"
            onClick={() => void send()}
            disabled={isSending}
          >
            {isSending ? "Đang gửi..." : "Gửi"}
          </button>
        </div>
      </section>
    </main>
  );
}

function FormattedMessage({ content }: { content: string }) {
  const lines = content.split("\n");
  return (
    <>
      {lines.map((line, lineIndex) => (
        <p
          className={lineIndex === 0 ? "m-0 whitespace-pre-wrap break-words leading-7" : "mt-[0.55rem] whitespace-pre-wrap break-words leading-7"}
          key={`${lineIndex}-${line}`}
        >
          {renderBoldSegments(line)}
        </p>
      ))}
    </>
  );
}

function renderBoldSegments(text: string) {
  const parts = text.split(/(\*\*.*?\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      return <strong key={`${index}-${part}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${index}-${part}`}>{part}</span>;
  });
}
