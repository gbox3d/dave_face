import { useMemo } from "react";
import qrcode from "qrcode-generator";

// 브라우저에서 만든다 — 밖으로 아무것도 요청하지 않는다 (WorkGate 의 Qr.tsx 와 같은 라이브러리).
export function Qr({ url, label }: { url: string; label: string }) {
  const svg = useMemo(() => {
    const qr = qrcode(0, "M");
    qr.addData(url, "Byte");
    qr.make();
    return qr.createSvgTag({ cellSize: 4, margin: 8, scalable: true });
  }, [url]);
  return <div className="qr-code" role="img" aria-label={label} dangerouslySetInnerHTML={{ __html: svg }} />;
}
