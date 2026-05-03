"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";

type Props = {
  mode: "login" | "register";
  onSubmit: (email: string, password: string) => Promise<void>;
};

export function AuthCard({ mode, onSubmit }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await onSubmit(email, password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-md rounded-3xl bg-card p-8 shadow-soft">
      <h1 className="text-2xl font-semibold text-ink">Traffic RAG Assistant</h1>
      <p className="mt-2 text-sm text-slate-500">
        {mode === "login" ? "Đăng nhập để tiếp tục hội thoại." : "Tạo tài khoản để bắt đầu."}
      </p>
      <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
        <input
          className="w-full rounded-xl border border-slate-200 px-4 py-3 outline-none ring-accent transition focus:ring-2"
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          className="w-full rounded-xl border border-slate-200 px-4 py-3 outline-none ring-accent transition focus:ring-2"
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error ? <p className="text-sm text-red-600">{error}</p> : null}
        <button
          className="w-full rounded-xl bg-accent px-4 py-3 font-medium text-white disabled:opacity-60"
          disabled={loading}
          type="submit"
        >
          {loading ? "Đang xử lý..." : mode === "login" ? "Đăng nhập" : "Đăng ký"}
        </button>
      </form>
      <p className="mt-4 text-sm text-slate-500">
        {mode === "login" ? "Chưa có tài khoản? " : "Đã có tài khoản? "}
        <Link className="font-medium text-accent" href={mode === "login" ? "/register" : "/login"}>
          {mode === "login" ? "Đăng ký" : "Đăng nhập"}
        </Link>
      </p>
    </div>
  );
}
