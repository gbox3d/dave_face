"""sqlite3 저장소: 사용자마다 등록 사진 한 장의 임베딩과 품질 값. 사진·크롭은 저장하지 않는다.

DB 파일과 관리자 토큰은 localfiles/ (git 밖) 에 0600 으로 둔다. 요청 스레드마다 연결을 새로 연다.
스키마 변경은 마이그레이션 없이 파일을 지우고 다시 만든다 (테스트 도구 — 등록은 다시 한다).
"""

import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  user_id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  embedding BLOB NOT NULL,                  -- 512 x float32 LE, L2 정규화 (등록 사진 한 장)
  eye_dist REAL NOT NULL, blur REAL NOT NULL, yaw REAL NOT NULL);
"""


def _private_file(path):
    """없으면 0600 으로 만든다."""
    os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o600))


class Store:
    def __init__(self, path):
        self.path = Path(path)
        _private_file(self.path)
        with self._db() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.path)
        try:
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn
        finally:
            conn.close()

    def replace_user(self, user_id, obs, now):
        """obs = emb, eye_dist, blur, yaw 를 가진 관측. 같은 ID 가 있었으면 바꾸고 True."""
        with self._db() as conn:
            existed = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,)).rowcount > 0
            conn.execute("INSERT INTO users(user_id, created_at, embedding, eye_dist, blur, yaw) VALUES (?, ?, ?, ?, ?, ?)",
                         (user_id, now, np.asarray(obs.emb, dtype="<f4").tobytes(),
                          float(obs.eye_dist), float(obs.blur), float(obs.yaw)))
        return existed

    def embeddings(self):
        """{user_id: 임베딩} — 인식 때 전부와 비교한다."""
        with self._db() as conn:
            rows = conn.execute("SELECT user_id, embedding FROM users ORDER BY user_id").fetchall()
        return {row["user_id"]: np.frombuffer(row["embedding"], dtype="<f4") for row in rows}

    def list_users(self):
        with self._db() as conn:
            rows = conn.execute("SELECT user_id, created_at, eye_dist, blur, yaw FROM users ORDER BY user_id").fetchall()
        return [dict(row) for row in rows]

    def delete_user(self, user_id):
        with self._db() as conn:
            return conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,)).rowcount > 0


def admin_token(path):
    """관리자 토큰을 읽는다. 없으면 만들어 0600 으로 저장한다 (WorkGate api-token 과 같은 방식).

    비었거나 짧은 토큰은 받지 않는다 — 빈 토큰은 헤더 없는 요청과 같아져 관리자 검사가 열린다.
    """
    path = Path(path)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
            handle.flush()
            os.fsync(handle.fileno())  # 전원이 나가도 0바이트 파일이 남지 않게
    token = path.read_text().strip()
    if len(token) < 32:
        raise RuntimeError(f"admin token in {path} is empty or too short; delete the file to regenerate it")
    return token
