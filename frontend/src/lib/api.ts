import { API_BASE } from "@/lib/config";
import type { AuthUser, ChatMessage, Thread } from "@/lib/types";

type RequestOptions = {
  method?: "GET" | "POST" | "PATCH";
  token?: string | null;
  body?: unknown;
};

async function api<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  };
  if (opts.token) {
    headers.Authorization = `Bearer ${opts.token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method: opts.method ?? "GET",
    headers,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    cache: "no-store"
  });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const payload = (await res.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }

  return (await res.json()) as T;
}

export async function register(email: string, password: string): Promise<{ access_token: string; user: AuthUser }> {
  return api("/auth/register", {
    method: "POST",
    body: { email, password }
  });
}

export async function login(email: string, password: string): Promise<{ access_token: string; user: AuthUser }> {
  return api("/auth/login", {
    method: "POST",
    body: { email, password }
  });
}

export async function me(token: string): Promise<{ user: AuthUser }> {
  return api("/auth/me", { token });
}

export async function createThread(token: string, title: string): Promise<Thread> {
  return api("/threads", {
    token,
    method: "POST",
    body: { title }
  });
}

export async function listThreads(token: string, includeArchived = false): Promise<{ threads: Thread[] }> {
  return api(`/threads?limit=50&include_archived=${includeArchived ? "true" : "false"}`, { token });
}

export async function patchThread(
  token: string,
  threadId: string,
  payload: { title?: string; archived?: boolean }
): Promise<{ thread: Thread }> {
  return api(`/threads/${threadId}`, {
    token,
    method: "PATCH",
    body: payload
  });
}

export async function listMessages(
  token: string,
  threadId: string,
  opts: { limit?: number; beforeMessageId?: string } = {}
): Promise<{ messages: ChatMessage[]; page: { limit: number; returned: number; next_before_message_id: string | null } }> {
  const params = new URLSearchParams();
  if (opts.limit) params.set("limit", String(opts.limit));
  if (opts.beforeMessageId) params.set("before_message_id", opts.beforeMessageId);
  const qs = params.toString();
  return api(`/threads/${threadId}/messages${qs ? `?${qs}` : ""}`, { token });
}

export async function chat(
  token: string,
  threadId: string,
  query: string
): Promise<{
  assistant_message: { content: string; citations?: { citation_map?: Record<string, unknown> } };
  context?: { confidence?: number };
}> {
  return api(`/threads/${threadId}/chat`, {
    token,
    method: "POST",
    body: {
      query,
      mode: "hybrid",
      dense_backend: "chroma",
      top_k: 5,
      candidate_k: 30,
      neighbor_window: 1,
      max_context_tokens: 1800
    }
  });
}
