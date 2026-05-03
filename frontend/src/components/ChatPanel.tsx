"use client";

import type { ChatMessage } from "@/lib/types";
import { FormEvent, useState } from "react";

type Props = {
  messages: ChatMessage[];
  loading: boolean;
  onSend: (query: string) => Promise<void>;
  onLoadMore: () => Promise<void>;
  canLoadMore: boolean;
};

export function ChatPanel({ messages, loading, onSend, onLoadMore, canLoadMore }: Props) {
  const [query, setQuery] = useState("");

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = query.trim();
    if (!text) return;
    setQuery("");
    await onSend(text);
  }

  return (
    <section className="flex h-full flex-1 flex-col rounded-3xl bg-card p-4 shadow-soft">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Conversation</h2>
        <button
          className="rounded-lg bg-slate-100 px-3 py-1 text-xs disabled:opacity-50"
          onClick={() => void onLoadMore()}
          disabled={!canLoadMore || loading}
        >
          Load older
        </button>
      </div>

      <div className="flex-1 space-y-3 overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50 p-4">
        {messages.map((msg) => (
          <div key={msg.id} className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm ${msg.role === "user" ? "ml-auto bg-accent text-white" : "bg-white text-ink"}`}>
            <p className="whitespace-pre-wrap">{msg.content}</p>
            <p className="mt-1 text-[11px] opacity-70">{new Date(msg.created_at).toLocaleTimeString()}</p>
          </div>
        ))}
      </div>

      <form className="mt-4 flex gap-3" onSubmit={handleSubmit}>
        <input
          className="flex-1 rounded-xl border border-slate-200 px-4 py-3 outline-none ring-accent transition focus:ring-2"
          placeholder="Hỏi về luật giao thông, GPLX..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="rounded-xl bg-accent px-5 py-3 text-white disabled:opacity-60" disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </form>
    </section>
  );
}
