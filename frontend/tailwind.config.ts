import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#f6f8fc",
        ink: "#102542",
        accent: "#1f6feb",
        card: "#ffffff"
      },
      boxShadow: {
        soft: "0 12px 36px rgba(16,37,66,0.08)"
      }
    }
  },
  plugins: []
};

export default config;
