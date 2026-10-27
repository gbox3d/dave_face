// 호스트 값은 아무것도 박지 않는다. 페이지는 자기 위치 기준 상대 경로로 API 를 부른다(api.ts).
// 그래서 같은 빌드가 nginx 접두사 뒤(/dave-face/)에서도, API 포트 직접(:3011/)에서도 돈다.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 페이지 셋: index.html(인식 테스트), compare.html(사진 비교), admin.html(등록·관리자). API 가 /, /compare, /admin 에서 서빙한다.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: { input: { index: "index.html", compare: "compare.html", admin: "admin.html" } },
  },
});
