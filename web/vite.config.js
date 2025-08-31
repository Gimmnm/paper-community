// web/vite.config.js
import react from '@vitejs/plugin-react'
import {defineConfig} from 'vite'

export default defineConfig({
  plugins: [react()],
  server: {port: 5173, open: true},
  // 若部署到子路径（比如 GitHub Pages），把 base 改成 '/paper-community/'
  base: '/'
})