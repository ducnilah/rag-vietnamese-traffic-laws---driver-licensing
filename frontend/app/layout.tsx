import "./globals.css";

export const metadata = {
  title: "Traffic Law Assistant",
  description: "Vietnamese traffic law RAG chatbot",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
