"use client";

import { AuthCard } from "@/components/AuthCard";
import { register } from "@/lib/api";
import { setToken } from "@/lib/auth";
import { useRouter } from "next/navigation";

export default function RegisterPage() {
  const router = useRouter();

  async function handleRegister(email: string, password: string) {
    const payload = await register(email, password);
    setToken(payload.access_token);
    router.replace("/chat");
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <AuthCard mode="register" onSubmit={handleRegister} />
    </main>
  );
}
