import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Traffic Law Assistant",
  description: "RAG chatbot for traffic law and driver licensing"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
