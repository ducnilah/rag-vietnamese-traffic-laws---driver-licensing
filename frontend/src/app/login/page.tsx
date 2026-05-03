"use client";

import { AuthCard } from "@/components/AuthCard";
import { setToken } from "@/lib/auth";
import { login } from "@/lib/api";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();

  async function handleLogin(email: string, password: string) {
    const payload = await login(email, password);
    setToken(payload.access_token);
    router.replace("/chat");
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <AuthCard mode="login" onSubmit={handleLogin} />
    </main>
  );
}
