import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/process-gdrive': {
        target: 'https://terminologically-fadlike-jaycee.ngrok-free.dev',
        changeOrigin: true,
        secure: false,
      },
      '/status': {
        target: 'https://terminologically-fadlike-jaycee.ngrok-free.dev',
        changeOrigin: true,
        secure: false,
      },
      '/download': {
        target: 'https://terminologically-fadlike-jaycee.ngrok-free.dev',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})