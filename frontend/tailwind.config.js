/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        muted: "var(--muted)",
        paper: "var(--paper)",
        panel: "var(--panel)",
        field: "var(--field)",
        line: "var(--line)",
        accent: "var(--accent)",
        "accent-strong": "var(--accent-strong)",
        "accent-warm": "var(--accent-warm)",
      },
      boxShadow: {
        panel: "var(--shadow)",
      },
      fontFamily: {
        serif: ['Georgia', '"Times New Roman"', "serif"],
      },
      backgroundImage: {
        "accent-primary": "linear-gradient(135deg, var(--accent), var(--accent-strong))",
      },
    },
  },
  plugins: [],
};
