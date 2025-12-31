/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class', // or 'media' if you prefer automatic OS dark mode
  content: [
    './index.html',
    './src/**/*.{ts,tsx,js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: '#0f1115',
          soft: '#171923',
          card: '#1a1e27'
        }
      },
      boxShadow: {
        soft: '0 6px 24px rgba(0,0,0,0.18)',
      },
      borderRadius: {
        xl2: '1.25rem',
      }
    },
  },
  plugins: [],
};
