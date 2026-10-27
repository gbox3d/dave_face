#!/usr/bin/env python3
"""접속 QR 카드(얼굴 인식 테스트 + 관리자)를 A4 PDF 로 만들어 IPP 프린터에 바로 보낸다.

  apps/api/.venv/bin/python scripts/print-qr.py              # 터널 주소·토큰을 읽어 만들고 인쇄
  apps/api/.venv/bin/python scripts/print-qr.py --no-print   # localfiles/qr/cards.pdf 만 만든다
  apps/api/.venv/bin/python scripts/print-qr.py --only demo  # 데모 카드만 (admin: 관리자 카드만)
  apps/api/.venv/bin/python scripts/print-qr.py --url https://host/dave-face/ --printer ipp://host:631/ipp/print

관리자 카드의 QR 에는 토큰이 URL 조각(#t=)으로 들어 있다 — 카드를 아무에게나 주지 말 것.
터널 주소는 터널이 재시작하면 바뀌므로 카드에 만든 시각을 찍는다. 한글 폰트가 없어 글자는 영문이다.
"""

import argparse
import datetime
import io
import os
import re
import subprocess
import sys
from pathlib import Path

import segno
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
TUNNEL_LOG = Path.home() / "works" / "baro_tunnel" / "localfiles" / "keeper-out.log"
TOKEN_FILE = ROOT / "localfiles" / "admin-token"
OUT = ROOT / "localfiles" / "qr" / "cards.pdf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DPI = 150
PAGE = (round(8.27 * DPI), round(11.69 * DPI))  # A4
CARDS = {  # 이름 -> (제목, 설명, 데모 주소 뒤에 붙는 경로)
    "demo": ("Dave Face - Face Recognition", ["Scan with your phone camera (HTTPS),", "then tap the start button."], ""),
    "admin": ("Dave Face - Admin", ["Opens the admin page with the token filled in.", "Keep this card private."], "admin"),
}


def tunnel_url():
    hosts = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", TUNNEL_LOG.read_text(errors="replace"))
    if not hosts:
        sys.exit(f"no tunnel address in {TUNNEL_LOG}")
    return hosts[-1] + "/dave-face/"


def find_printer():
    found = subprocess.run(["ippfind", "-T", "5"], capture_output=True, text=True).stdout.split()
    if not found:
        sys.exit("no IPP printer found (ippfind)")
    return found[0]


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size)


def card(title, lines, qr_url, shown_url, stamp):
    page = Image.new("RGB", PAGE, "white")
    draw = ImageDraw.Draw(page)
    width, height = PAGE
    buffer = io.BytesIO()
    segno.make(qr_url, error="m").save(buffer, kind="png", scale=10, border=2)
    side = round(width * 0.6)
    page.paste(Image.open(buffer).convert("RGB").resize((side, side), Image.NEAREST), ((width - side) // 2, round(height * 0.2)))
    draw.text((width / 2, height * 0.12), title, fill="black", font=font(64), anchor="mm")
    y = height * 0.2 + side + 60
    for line in lines:
        draw.text((width / 2, y), line, fill="black", font=font(34), anchor="mm")
        y += 50
    draw.text((width / 2, y + 30), shown_url, fill="#333333", font=font(26), anchor="mm")
    draw.text((width / 2, height - 80), f"Printed {stamp} - the address changes when the tunnel restarts",
              fill="#666666", font=font(22), anchor="mm")
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="base URL of the demo page (default: latest tunnel address + /dave-face/)")
    parser.add_argument("--token-file", type=Path, default=TOKEN_FILE)
    parser.add_argument("--only", choices=sorted(CARDS), help="just one card")
    parser.add_argument("--printer", help="IPP URI (default: the first printer ippfind sees)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-print", action="store_true", help="only write the PDF")
    args = parser.parse_args()

    base = args.url or tunnel_url()
    if not base.endswith("/"):
        base += "/"
    token = args.token_file.read_text().strip()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pages = []
    for name in [args.only] if args.only else list(CARDS):
        title, lines, path = CARDS[name]
        shown = base + path
        pages.append(card(title, lines, shown + (f"#t={token}" if name == "admin" else ""), shown, stamp))

    args.out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # 관리자 카드에 토큰이 들어 있다
    with os.fdopen(fd, "wb") as handle:
        pages[0].save(handle, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print(f"wrote {args.out} ({len(pages)} page(s)) for {base}")
    if args.no_print:
        return
    printer = args.printer or find_printer()
    print(f"printing to {printer}")
    result = subprocess.run(["ipptool", "-t", "-f", str(args.out), "-d", "filetype=application/pdf", printer,
                             "print-job.test"], capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        sys.exit(result.stderr or f"ipptool exit {result.returncode}")


if __name__ == "__main__":
    main()
