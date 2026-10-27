// 사진 비교 화면: 기준 사진 한 장을 찍고, 카메라에 비치는 얼굴과의 유사도를 바로 본다.
// 등록·토큰·저장 없음 — 서버(/api/embed)는 임베딩만 돌려주고, 코사인(두 단위 벡터의 내적)은 여기서 잰다.
import { useEffect, useRef, useState } from "react";
import { ApiError, type EmbedReply, type Face, embed } from "./api";
import { CANCELLED, HINT, Page, Stage, TimingLine, type Timing, addTiming, describe, faceText, newTiming, useCamera,
         useClearOnRestore, useHealth, wait } from "./flow";

interface Reference { embedding: number[]; photo: string; face: Face }

const dot = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0);

export function Compare() {
  const health = useHealth();
  const [running, setRunning] = useState(false);
  const [reference, setReference] = useState<Reference | null>(null);
  const [live, setLive] = useState<EmbedReply | null>(null);
  const [score, setScore] = useState<number | null>(null);
  const [timing, setTiming] = useState<Timing | null>(null);
  const [error, setError] = useState("");
  const [capturing, setCapturing] = useState(false);
  const wantReference = useRef(false);
  const current = useRef<Reference | null>(null);  // 루프는 state 대신 이것을 읽는다 (클로저는 옛 state 를 쥔다)
  const { video, cancelled, withCamera } = useCamera();
  useClearOnRestore(() => setError(""));

  useEffect(() => () => { if (current.current) URL.revokeObjectURL(current.current.photo); }, []);

  function keepReference(next: Reference) {
    if (current.current) URL.revokeObjectURL(current.current.photo);
    current.current = next;
    setReference(next);
  }

  // 다음에 얼굴이 잡힌 프레임이 기준이 된다.
  function takeReference() {
    wantReference.current = true;
    setCapturing(true);
  }

  async function start() {
    setError("");
    setLive(null);
    setScore(null);
    setTiming(null);
    setRunning(true);
    try {
      await withCamera(async (grab) => {
        let t = newTiming();
        for (;;) {  // 정지를 누르면 grab() 이 CANCELLED 를 던져 빠져나온다
          const jpeg = await grab();
          const armed = wantReference.current;  // 탭한 뒤에 찍힌 프레임만 기준이 된다
          const sent = performance.now();
          let r: EmbedReply;
          try {
            r = await embed(jpeg);
          } catch (e) {
            if (e instanceof ApiError && e.status === 503) { await wait(200); continue; }
            throw e;
          }
          t = addTiming(t, r, performance.now() - sent);
          setTiming(t);
          setLive(r);
          if (!r.embedding || !r.face) { setScore(null); continue; }
          if (armed) {
            wantReference.current = false;
            setCapturing(false);
            keepReference({ embedding: r.embedding, photo: URL.createObjectURL(jpeg), face: r.face });
          }
          setScore(current.current ? dot(current.current.embedding, r.embedding) : null);
        }
      });
    } catch (e) {
      if (!(e instanceof Error && e.message === CANCELLED)) setError(describe(e));
    } finally {
      wantReference.current = false;
      setCapturing(false);
      setRunning(false);
    }
  }

  const threshold = live?.threshold ?? 0.5;
  const same = score !== null && score >= threshold;
  let headline = "—";
  if (score !== null) headline = score.toFixed(3);
  else if (live && live.status !== "ok") headline = HINT[live.status] ?? live.status;
  else if (!reference) headline = "기준 사진을 찍으세요";

  return (
    <Page page="compare" eyebrow="사진 비교 테스트" health={health}>
      <Stage video={video} on={running} />

      <section className="panel">
        {running ? (
          <div className="row">
            <button type="button" disabled={capturing} onClick={takeReference}>
              {capturing ? "얼굴을 찾는 중…" : reference ? "기준 다시 찍기" : "기준 사진 찍기"}
            </button>
            <button type="button" className="ghost" onClick={() => { cancelled.current = true; }}>정지</button>
          </div>
        ) : (
          <button type="button" className="big" disabled={!health?.ok} onClick={start}>카메라 켜기</button>
        )}
        {!running && (
          <p className="note">
            기준 사진 한 장을 찍으면, 그 뒤로 카메라에 비치는 얼굴과의 유사도(코사인)를 실시간으로 보여 줍니다.
            등록·저장 없이 즉석에서 비교합니다 — 다른 사람이나 다른 거리·각도로 비춰 보세요.
          </p>
        )}
        {error && <p className="admin-error" role="alert">{error}</p>}
      </section>

      {(reference || live) && (
        <section className={`panel result ${score === null ? "" : same ? "ok" : "fail"}`} aria-live="polite">
          <div className="compare-head">
            {reference && <img className="thumb" src={reference.photo} alt="기준 사진" />}
            <div>
              <h2 className={score !== null ? "score" : ""}>{headline}</h2>
              {score !== null && <p>{same ? "같은 사람" : "다른 사람"} · 문턱 {threshold}</p>}
            </div>
          </div>
          {score !== null && (
            <div className={`bar ${same ? "hit" : ""}`}><div style={{ width: `${Math.min(100, Math.max(0, score) * 100)}%` }} /></div>
          )}
          {reference && <p className="meta">기준 · {faceText(reference.face)}</p>}
          {live?.face && <p className="meta">지금 · {faceText(live.face)}</p>}
          {timing && <TimingLine t={timing} />}
        </section>
      )}
    </Page>
  );
}
