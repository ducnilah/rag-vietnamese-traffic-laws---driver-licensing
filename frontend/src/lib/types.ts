export type AuthUser = {
  id: string;
  email: string | null;
  is_active: boolean;
  created_at: string;
};

export type Thread = {
  id: string;
  user_id: string;
  title: string;
  archived: boolean;
  created_at: string;
  last_active_at: string;
};

export type ChatMessage = {
  id: string;
  thread_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Record<string, unknown> | null;
  created_at: string;
};
