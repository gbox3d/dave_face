const path = require("node:path");
const root = path.resolve(__dirname, "..");

module.exports = {
  apps: [{
    name: "dave-face",
    cwd: root,
    script: path.join(root, "apps", "api", "run-web.sh"),
    interpreter: "/bin/bash",
    exec_mode: "fork",
    instances: 1, // One Gunicorn master; its single worker owns the CUDA context + TensorRT engines (no --preload).
    autorestart: true,
    restart_delay: 5000,
    // run-web.sh 의 사전 검사(엔진 없음·GPU 를 oni-detector 가 점유·포트 선점)가 exit 2 로 끝나면 무한 재시작 대신 errored 로 멈춘다.
    exp_backoff_restart_delay: 5000,
    max_restarts: 10,
    min_uptime: 30000,
    kill_timeout: 20000,
    env: {
      PYTHONUNBUFFERED: "1",
      DAVE_FACE_PORT: "3011",
      CUDA_MODULE_LOADING: "LAZY", // CUDA 12.2 기본값이지만 명시한다 (커널 모듈 지연 로딩 — 메모리·시작 시간)
    },
  }],
};
