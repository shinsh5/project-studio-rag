import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  build: {
    outDir: 'app/static/pipeline-flow',
    emptyOutDir: true,
    sourcemap: false,
    lib: {
      entry: 'frontend/pipeline-flow.jsx',
      formats: ['es'],
      fileName: () => 'pipeline-flow.js',
    },
    rollupOptions: {
      output: {
        assetFileNames: 'pipeline-flow.[ext]',
      },
    },
  },
});
