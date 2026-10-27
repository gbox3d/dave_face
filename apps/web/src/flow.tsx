// 테스트(Recognize)와 관리자(Admin) 페이지가 같이 쓰는 것: 안내 문구, health 폴링, 카메라 흐름, 공통 레이아웃.
import { type ReactNode, type RefObject, useEffect, useRef, useState } from "react";
import { ApiError, type Face, type Health, fetchHealth } from "./api";
import { captureJpeg, openCamera, stopCamera } from "./capture";

export const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
export const CANCELLED = "cancelled";

// 얼굴을 못 찾았을 때 안내
export const HINT: Record<string, string> = { no_face: "얼굴이 보이지 않아요 — 카메라를 보세요" };

export const faceText = (f: Face) =>
  `눈 거리 ${f.eye_dist}px · yaw ${f.yaw} · roll ${f.roll}° · 선명도 ${f.blur}`;

export function describe(error: unknown): string {
  if (error instanceof ApiError) {
    const body = error.body ?? {};
    const code = typeof body.error === "string" ? body.error : "";
    if (HINT[code]) return HINT[code];
    switch (code) {
      case "frame_missing": return "사진이 없습니다";
      case "invalid_frame": return "사진 형식이 맞지 않습니다";
      case "invalid_user_id": return "ID 는 영문 소문자·숫자·_·- 로 1~32자입니다";
      case "unauthorized": return "관리자 토큰이 맞지 않습니다";
      case "unavailable": case "busy": return "얼굴 인식 엔진이 준비되지 않았습니다. 잠시 후 다시 시도하세요";
    }
    return `요청이 실패했습니다 (HTTP ${error.status})`;
  }
  if (error instanceof DOMException && error.name === "NotAllowedError") return "카메라 권한을 허용해 주세요";
  if (error instanceof DOMException && error.name === "TimeoutError") return "서버 응답이 없습니다";
  if (error instanceof Error && error.message === "insecure") return "카메라는 HTTPS(보안 연결) 주소에서만 열립니다";
  if (error instanceof TypeError) return `서버와 연결이 끊겼습니다 (${error.message})`;  // fetch 네트워크 오류
  return error instanceof Error ? error.message : String(error);
}

// iOS Safari 는 다른 화면으로 넘어가는 순간 진행 중인 요청을 끊고("Load failed") 이 페이지를 뒤로 가기 캐시에 넣는다.
// 뒤로 돌아왔을 때 그 오류가 남아 보이지 않게, 캐시에서 되살아나면 지운다.
export function useClearOnRestore(clear: () => void) {
  useEffect(() => {
    const onShow = (event: PageTransitionEvent) => { if (event.persisted) clear(); };
    window.addEventListener("pageshow", onShow);
    return () => window.removeEventListener("pageshow", onShow);
  }, []);
}

export function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);
  useEffect(() => {
    let timer = 0;
    let alive = true;
    const tick = async () => {
      const next = await fetchHealth().catch(() => null);
      if (!alive) return;
      setHealth(next);
      timer = window.setTimeout(tick, next?.ok ? 10000 : 2000);
    };
    void tick();
    return () => { alive = false; clearTimeout(timer); };
  }, []);
  return health;
}

export function useCamera() {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const cancelled = useRef(false);

  async function withCamera<T>(run: (grab: () => Promise<Blob>) => Promise<T>): Promise<T> {
    const element = video.current!;
    cancelled.current = false;
    const stream = await openCamera(element);
    try {
      canvas.current ??= document.createElement("canvas");
      return await run(() => {
        if (cancelled.current) throw new Error(CANCELLED);
        return captureJpeg(element, canvas.current!);
      });
    } finally {
      stopCamera(stream, element);
    }
  }

  return { video, cancelled, withCamera };
}

// 응답 성능: 서버가 잰 처리 시간(elapsed_ms)·NPU 시간(npu_ms)과 폰이 잰 왕복을 프레임마다 더한다.
// fps 는 첫 프레임 요청부터 마지막 응답까지로 잰다 — 정지한 뒤 다시 그려져도 값이 바뀌지 않는다.
export interface Timing { frames: number; server: number; npu: number; rtt: number; started: number; last: number }
export function newTiming(): Timing {
  const now = performance.now();
  return { frames: 0, server: 0, npu: 0, rtt: 0, started: now, last: now };
}
export const addTiming = (t: Timing, r: { elapsed_ms: number; npu_ms: number }, rtt: number): Timing =>
  ({ ...t, frames: t.frames + 1, server: t.server + r.elapsed_ms, npu: t.npu + r.npu_ms, rtt: t.rtt + rtt,
     last: performance.now() });

export function TimingLine({ t }: { t: Timing }) {
  if (!t.frames) return null;
  const avg = (sum: number) => Math.round(sum / t.frames);
  const seconds = (t.last - t.started) / 1000;
  return (
    <p className="timing">
      <span>{t.frames}프레임 / {seconds.toFixed(0)}s ({(t.frames / Math.max(seconds, 0.1)).toFixed(1)} fps)</span>
      <span>서버 {avg(t.server)} ms</span>
      <span>NPU {avg(t.npu)} ms</span>
      <span>왕복 {avg(t.rtt)} ms</span>
    </p>
  );
}

// 화면 셋을 잇는 메뉴. 상대 경로라 /dave-face/ 접두사 뒤에서도, :3011 직접에서도 맞다.
const NAV = [["test", "./", "인식 테스트"], ["compare", "compare", "사진 비교"], ["admin", "admin", "등록 (관리자)"]] as const;

export function Page({ page, eyebrow, health, children }:
                     { page: (typeof NAV)[number][0]; eyebrow: string; health: Health | null; children: ReactNode }) {
  const ready = health?.ok ?? false;
  return (
    <main className="app">
      <header>
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>Dave Face</h1>
        </div>
        <span className={`pill ${ready ? "ok" : "wait"}`}>{health ? (ready ? "준비됨" : health.state) : "연결 중"}</span>
      </header>
      <nav className="nav">
        {NAV.map(([key, href, label]) => (
          <a key={key} href={href} aria-current={key === page ? "page" : undefined}>{label}</a>
        ))}
      </nav>
      {children}
      <footer>
        <span>보안 연결 {window.isSecureContext ? "예" : "아니요 (카메라 불가)"}</span>
        {health && <span>v{health.version}</span>}
      </footer>
    </main>
  );
}

export function Stage({ video, on }: { video: RefObject<HTMLVideoElement | null>; on: boolean }) {
  return (
    <section className={`camera ${on ? "on" : ""}`} aria-hidden={!on}>
      <video ref={video} playsInline muted />
      <div className="guide" />
    </section>
  );
}
