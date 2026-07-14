/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        cinema: {
          900: '#0a0a0f',
          800: '#12121a',
          700: '#1a1a2e',
          600: '#2a2a3e',
          500: '#3a3a4e',
          400: '#6a6a7e',
          300: '#9a9aae',
          200: '#cacade',
          100: '#eaeafe',
        },
        accent: {
          DEFAULT: '#e50914',
          hover: '#f40612',
          muted: '#831010',
        },
        gold: {
          DEFAULT: '#f5c518',
          dark: '#c29b12',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
