const path = require("node:path");
const root = path.resolve(__dirname, "..");

module.exports = {
  apps: [{
    name: "dave-face",
    cwd: root,
    script: path.join(root, "apps", "api", "run-web.sh"),
    interpreter: "/bin/bash",
    exec_mode: "fork",
    instances: 1, // One Gunicorn master owns the Hailo device.
    autorestart: true,
    restart_delay: 5000,
    kill_timeout: 20000,
    env: {
      PYTHONUNBUFFERED: "1",
      DAVE_FACE_PORT: "3011",
      HAILORT_LOGGER_PATH: path.join(root, "localfiles", "logs"),
    },
  }],
};
