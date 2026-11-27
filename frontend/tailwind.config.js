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
          950: 'var(--pacs-950)',
          900: 'var(--pacs-900)',
          850: 'var(--pacs-850)',
          800: 'var(--pacs-800)',
          700: 'var(--pacs-700)',
          600: 'var(--pacs-600)',
          500: 'var(--pacs-500)',
          400: 'var(--pacs-400)',
          300: 'var(--pacs-300)',
          200: 'var(--pacs-200)',
          100: 'var(--pacs-100)',
        },
        clinical: {
          cyan: '#06B6D4',
          sky: '#38BDF8',
          teal: '#14B8A6',
          green: '#10B981',
          amber: '#F59E0B',
          rose: '#F43F5E',
          purple: '#8B5CF6',
          indigo: '#6366F1',
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

