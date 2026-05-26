/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          DEFAULT: '#1b2a4a',
          light: '#2c3e63',
          dark: '#121d36',
        },
        cream: {
          DEFAULT: '#f7f3e9',
          dark: '#ece5d3',
        },
      },
      fontFamily: {
        serif: ['Georgia', 'Cambria', 'Times New Roman', 'serif'],
      },
    },
  },
  plugins: [],
}
