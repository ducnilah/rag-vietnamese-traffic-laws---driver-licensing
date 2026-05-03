"use client";

import { ChatPanel } from "@/components/ChatPanel";
import { ThreadSidebar } from "@/components/ThreadSidebar";
import { clearToken, getToken } from "@/lib/auth";
import { chat, createThread, listMessages, listThreads, me, patchThread } from "@/lib/api";
import type { ChatMessage, Thread } from "@/lib/types";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";

export default function ChatPage() {
  const router = useRouter();
  const [token, setTokenState] = useState<string | null>(null);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);

  useEffect(() => {
    const t = getToken();
    if (!t) {
      router.replace("/login");
      return;
    }
    setTokenState(t);
  }, [router]);

  useEffect(() => {
    if (!token) return;
    void bootstrap(token);
  }, [token]);

  async function bootstrap(t: string) {
    try {
      await me(t);
      const threadResp = await listThreads(t, false);
      setThreads(threadResp.threads);
      const first = threadResp.threads[0];
      if (first) {
        setActiveThreadId(first.id);
        await loadMessages(t, first.id);
      }
    } catch {
      clearToken();
      router.replace("/login");
    }
  }

  async function loadMessages(t: string, threadId: string, beforeMessageId?: string) {
    const resp = await listMessages(t, threadId, { limit: 20, beforeMessageId });
    if (!beforeMessageId) {
      setMessages(resp.messages);
    } else {
      setMessages((prev) => [...resp.messages, ...prev]);
    }
    setNextCursor(resp.page.next_before_message_id);
  }

  async function handleCreate() {
    if (!token) return;
    const t = await createThread(token, "New chat");
    const next = [t, ...threads];
    setThreads(next);
    setActiveThreadId(t.id);
    setMessages([]);
    setNextCursor(null);
  }

  async function handleSelect(threadId: string) {
    if (!token) return;
    setActiveThreadId(threadId);
    await loadMessages(token, threadId);
  }

  async function handleRename(threadId: string) {
    if (!token) return;
    const title = window.prompt("Nhập title mới:");
    if (!title) return;
    const resp = await patchThread(token, threadId, { title });
    setThreads((prev) => prev.map((row) => (row.id === threadId ? resp.thread : row)));
  }

  async function handleArchive(threadId: string) {
    if (!token) return;
    await patchThread(token, threadId, { archived: true });
    const remaining = threads.filter((row) => row.id !== threadId);
    setThreads(remaining);
    if (activeThreadId === threadId) {
      setActiveThreadId(remaining[0]?.id ?? null);
      if (remaining[0]) {
        await loadMessages(token, remaining[0].id);
      } else {
        setMessages([]);
      }
    }
  }

  async function handleSend(query: string) {
    if (!token || !activeThreadId) return;
    setLoading(true);
    try {
      await chat(token, activeThreadId, query);
      await loadMessages(token, activeThreadId);
      const threadResp = await listThreads(token, false);
      setThreads(threadResp.threads);
    } finally {
      setLoading(false);
    }
  }

  async function handleLoadMore() {
    if (!token || !activeThreadId || !nextCursor) return;
    await loadMessages(token, activeThreadId, nextCursor);
  }

  const canLoadMore = useMemo(() => Boolean(nextCursor), [nextCursor]);

  function handleLogout() {
    clearToken();
    router.replace("/login");
  }

  return (
    <main className="h-screen p-4 md:p-6">
      <div className="mb-3 flex items-center justify-between rounded-2xl bg-white/80 px-4 py-3 shadow-soft backdrop-blur">
        <h1 className="text-lg font-semibold text-ink">Traffic Law & Driver Licensing Assistant</h1>
        <button className="rounded-lg bg-slate-100 px-3 py-2 text-sm" onClick={handleLogout}>
          Logout
        </button>
      </div>
      <div className="flex h-[calc(100vh-90px)] gap-4">
        <ThreadSidebar
          threads={threads}
          activeThreadId={activeThreadId}
          onCreate={handleCreate}
          onSelect={handleSelect}
          onRename={handleRename}
          onArchive={handleArchive}
        />
        <ChatPanel
          messages={messages}
          loading={loading}
          onSend={handleSend}
          onLoadMore={handleLoadMore}
          canLoadMore={canLoadMore}
        />
      </div>
    </main>
  );
}
