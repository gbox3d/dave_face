// 얼굴 메시 오버레이: MediaPipe Face Landmarker(브라우저 안 WASM·WebGL)가 카메라 영상에서 얼굴 점 478개를 찾고 영상 위 캔버스에 그린다.
// 서버 인식과는 무관한 장식이라 WASM·모델을 못 받거나 검출이 실패해도 인식 루프는 그대로 돈다.
// 패키지는 카메라를 켤 때 동적 import 로 따로 받는다(별도 청크) — 관리자 번들은 이것을 내려받지 않는다.
import { type RefObject, useEffect, useRef } from "react";
import type { DrawingUtils, FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

export type Tone = "" | "ok" | "fail";

// styles.css 다크 테마의 --accent·--ok·--fail 과 같은 값 — 매 프레임 CSS 변수를 읽지 않으려고 박아 둔다.
const COLOR: Record<Tone, string> = { "": "#4aa3ff", ok: "#4cc38a", fail: "#ff8a80" };

type Vision = typeof import("@mediapipe/tasks-vision");
export interface Mesh { vision: Vision; landmarker: FaceLandmarker }

let loading: Promise<Mesh> | undefined;

// WASM(약 11 MB, 첫 번만)·모델은 한 번 받아 세션 내내 쓴다 (close 하지 않는다 — 다음 실행이 바로 이어 쓴다).
// GPU(WebGL) 가 안 되면 CPU 로 한 번 더 시도하고, 그래도 안 되면 거부된 약속을 남겨 이 세션에서는 메시를 끈다 (경고는 한 번).
// 경로는 api.ts 처럼 문서 주소 기준 — /dave-face/ 접두사 뒤에서도, :3011 직접에서도 맞다.
export function loadLandmarker(): Promise<Mesh> {
  return (loading ??= (async () => {
    const vision = await import("@mediapipe/tasks-vision");
    const fileset = await vision.FilesetResolver.forVisionTasks(new URL("assets/mediapipe/wasm", document.baseURI).href);
    const create = (delegate: "GPU" | "CPU") => vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: new URL("assets/mediapipe/face_landmarker.task", document.baseURI).href, delegate },
      runningMode: "VIDEO",
      numFaces: 1,
    });
    const landmarker = await create("GPU").catch(() => create("CPU"));
    return { vision, landmarker };
  })().catch((e: unknown) => {
    console.warn("얼굴 메시를 끕니다 — MediaPipe 를 불러오지 못했습니다", e);
    throw e;
  }));
}

// on 인 동안 매 애니메이션 프레임 캔버스를 다시 그린다(맥박이 움직이게). 검출은 영상 프레임이 바뀐 때만 — 같은 프레임을 두 번 돌리지 않는다.
// 캔버스는 영상의 원본 크기로 맞춘다: CSS(object-fit: cover + 거울)가 영상과 같아서 정규화 좌표를 그대로 그리면 겹친다.
export function useFaceMesh(video: RefObject<HTMLVideoElement | null>, canvas: RefObject<HTMLCanvasElement | null>,
                            on: boolean, tone: Tone) {
  const toneRef = useRef(tone);
  useEffect(() => { toneRef.current = tone; }, [tone]);  // 색만 바뀔 때는 루프를 다시 돌리지 않는다

  useEffect(() => {
    const element = video.current;
    const ctx = canvas.current?.getContext("2d");
    if (!on || !element || !ctx) return;
    let frame = 0;
    let mesh: Mesh | null = null;
    let utils: DrawingUtils | null = null;
    let faces: NormalizedLandmark[][] = [];
    let seen = -1;  // 마지막으로 검출한 영상 시각(currentTime) — rVFC 가 없는 브라우저에서만 쓴다
    // 실패는 loadLandmarker 가 이미 경고했다 — 오버레이 없이 간다
    loadLandmarker().then((m) => { mesh = m; utils = new m.vision.DrawingUtils(ctx); }, () => {});

    // 새 영상 프레임은 requestVideoFrameCallback(Safari 15.4+·Chrome 83+·Firefox 132+)이 알리고, 없는 브라우저만 currentTime 변화로 가늠한다.
    // Safari 는 MediaStream 영상의 currentTime 을 벽시계로 주어 매 tick 바뀌므로(검출이 카메라 30 Hz 가 아니라 화면 주사율로 돈다) rVFC 가 꼭 필요하다.
    const rvfc = typeof element.requestVideoFrameCallback === "function";
    let fresh = false;  // rVFC 가 알린 새 프레임을 아직 검출하지 않았다
    let vfc = 0;
    const onFrame = () => { fresh = true; vfc = element.requestVideoFrameCallback(onFrame); };
    if (rvfc) vfc = element.requestVideoFrameCallback(onFrame);

    // 그리기를 멈추고 캔버스를 비운다 — 오류로 그만둘 때도, on 이 꺼질 때도 같다
    const stop = () => {
      cancelAnimationFrame(frame);
      if (rvfc) element.cancelVideoFrameCallback(vfc);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    };

    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      const { videoWidth: w, videoHeight: h } = element;
      if (!w) return;  // 카메라가 첫 프레임을 주기 전
      const surface = ctx.canvas;
      if (surface.width !== w || surface.height !== h) { surface.width = w; surface.height = h; }  // 크기를 넣으면 지워지므로 바뀔 때만
      try {
        if (mesh && (rvfc ? fresh : element.currentTime !== seen)) {
          fresh = false;
          seen = element.currentTime;
          faces = mesh.landmarker.detectForVideo(element, performance.now()).faceLandmarks;
        }
        ctx.clearRect(0, 0, w, h);
        const face = faces[0];
        if (!mesh || !utils || !face) return;
        const { FaceLandmarker: FL } = mesh.vision;
        const color = COLOR[toneRef.current];
        const pulse = 0.55 + 0.25 * Math.sin(now / 350);
        ctx.shadowBlur = 0;
        ctx.globalAlpha = pulse * 0.35;  // 테셀레이션은 연하게
        utils.drawConnectors(face, FL.FACE_LANDMARKS_TESSELATION, { color, lineWidth: 0.6 });
        // 번짐(shadow)은 윤곽·홍채에만 — 테셀레이션 2,556 선마다 그림자를 주면 폰에서 너무 느리다
        ctx.globalAlpha = pulse;
        ctx.shadowBlur = 10;
        ctx.shadowColor = color;
        for (const part of [FL.FACE_LANDMARKS_CONTOURS, FL.FACE_LANDMARKS_LEFT_IRIS, FL.FACE_LANDMARKS_RIGHT_IRIS]) {
          utils.drawConnectors(face, part, { color, lineWidth: 1.5 });
        }
      } catch (e) {
        stop();  // 이번 실행에서는 그만 그린다(남은 메시도 지운다) — 인식 루프는 영향받지 않는다
        console.warn("얼굴 메시를 끕니다 — 검출·그리기 오류", e);
      }
    };
    frame = requestAnimationFrame(tick);
    return stop;
  }, [on]);
}
