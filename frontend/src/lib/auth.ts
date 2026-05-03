import { STORAGE_TOKEN_KEY } from "@/lib/config";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(STORAGE_TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(STORAGE_TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(STORAGE_TOKEN_KEY);
}
