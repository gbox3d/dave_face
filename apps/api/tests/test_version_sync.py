"""WorkGate 는 /api/version 을 저장소 버전과 비교한다 — 버전 문자열 네 곳이 같아야 한다."""

import json
import tomllib
import unittest
from pathlib import Path

from dave_face import __version__

ROOT = Path(__file__).resolve().parents[3]


class VersionSyncTest(unittest.TestCase):
    def test_all_versions_match(self):
        versions = {
            "package.json": json.loads((ROOT / "package.json").read_text())["version"],
            "apps/web/package.json": json.loads((ROOT / "apps/web/package.json").read_text())["version"],
            "apps/api/pyproject.toml": tomllib.loads((ROOT / "apps/api/pyproject.toml").read_text())["project"]["version"],
            "dave_face.__version__": __version__,
        }
        self.assertEqual(len(set(versions.values())), 1, versions)


if __name__ == "__main__":
    unittest.main()
