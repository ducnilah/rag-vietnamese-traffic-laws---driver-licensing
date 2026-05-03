"use client";

import type { Thread } from "@/lib/types";

type Props = {
  threads: Thread[];
  activeThreadId: string | null;
  onCreate: () => Promise<void>;
  onSelect: (threadId: string) => void;
  onRename: (threadId: string) => Promise<void>;
  onArchive: (threadId: string) => Promise<void>;
};

export function ThreadSidebar({ threads, activeThreadId, onCreate, onSelect, onRename, onArchive }: Props) {
  return (
    <aside className="flex h-full w-[320px] flex-col rounded-3xl bg-card p-4 shadow-soft">
      <button className="rounded-xl bg-ink px-4 py-3 text-sm font-medium text-white" onClick={() => void onCreate()}>
        + New chat
      </button>
      <div className="mt-4 flex-1 space-y-2 overflow-y-auto">
        {threads.map((thread) => (
          <div
            key={thread.id}
            className={`rounded-xl border p-3 ${activeThreadId === thread.id ? "border-accent bg-blue-50" : "border-slate-200 bg-white"}`}
          >
            <button className="w-full text-left text-sm font-medium" onClick={() => onSelect(thread.id)}>
              {thread.title}
            </button>
            <p className="mt-1 text-xs text-slate-400">{new Date(thread.last_active_at).toLocaleString()}</p>
            <div className="mt-2 flex gap-2">
              <button className="rounded-lg bg-slate-100 px-2 py-1 text-xs" onClick={() => void onRename(thread.id)}>
                Rename
              </button>
              <button className="rounded-lg bg-rose-50 px-2 py-1 text-xs text-rose-700" onClick={() => void onArchive(thread.id)}>
                Archive
              </button>
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
