/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        pacs: {
          950: '#070B14',
          900: '#0B1120',
          850: '#0F172A',
          800: '#152033',
          700: '#1E293B',
          600: '#334155',
          500: '#475569',
          400: '#64748B',
          300: '#94A3B8',
          200: '#CBD5E1',
          100: '#E2E8F0',
        },
        clinical: {
          cyan: '#38BDF8',
          teal: '#2DD4BF',
          green: '#22C55E',
          amber: '#F59E0B',
          rose: '#F43F5E',
          purple: '#A855F7',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
