import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dave_face import db


def template(seed, yaw=0.01):
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return SimpleNamespace(emb=v / np.linalg.norm(v), eye_dist=120.0, blur=500.0, yaw=yaw)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "dave_face.db"
        self.store = db.Store(self.path)

    def test_file_is_private(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_one_embedding_per_user(self):
        self.assertFalse(self.store.replace_user("p1", template(1), now=1.0))
        self.assertTrue(self.store.replace_user("p1", template(2, yaw=0.05), now=2.0))  # 같은 ID 는 교체
        self.store.replace_user("p2", template(3), now=3.0)
        embeddings = self.store.embeddings()
        self.assertEqual(list(embeddings), ["p1", "p2"])
        np.testing.assert_allclose(embeddings["p1"], template(2).emb, rtol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(embeddings["p1"])), 1.0, places=5)
        self.assertEqual(self.store.list_users(),
                         [{"user_id": "p1", "created_at": 2.0, "eye_dist": 120.0, "blur": 500.0, "yaw": 0.05},
                          {"user_id": "p2", "created_at": 3.0, "eye_dist": 120.0, "blur": 500.0, "yaw": 0.01}])

    def test_delete(self):
        self.assertEqual(self.store.embeddings(), {})
        self.store.replace_user("p1", template(1), now=1.0)
        self.assertTrue(self.store.delete_user("p1"))
        self.assertFalse(self.store.delete_user("p1"))
        self.assertEqual(self.store.list_users(), [])


class AdminTokenTest(unittest.TestCase):
    def test_created_once_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "admin-token"
            first = db.admin_token(path)
            self.assertGreaterEqual(len(first), 40)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(db.admin_token(path), first)

    def test_empty_or_short_token_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "admin-token"
            for content in ("", "\n", "short-token"):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(RuntimeError):
                        db.admin_token(path)


if __name__ == "__main__":
    unittest.main()
