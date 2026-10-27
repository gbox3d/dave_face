// 테스트 페이지: 카메라 프레임마다 등록된 모든 사람과의 유사도·문턱·얼굴 값·처리 시간을 그대로 보여 준다.
// 세션도, 동작 지시도 없다 — 인식 성능을 눈으로 보는 화면이다.
import { useState } from "react";
import { ApiError, type RecognizeReply, recognize } from "./api";
import { CANCELLED, HINT, Page, Stage, TimingLine, type Timing, addTiming, describe, faceText, newTiming, useCamera,
         useClearOnRestore, useHealth, wait } from "./flow";

export function Recognize() {
  const health = useHealth();
  const [running, setRunning] = useState(false);
  const [reply, setReply] = useState<RecognizeReply | null>(null);
  const [timing, setTiming] = useState<Timing | null>(null);
  const [error, setError] = useState("");
  const { video, cancelled, withCamera } = useCamera();
  useClearOnRestore(() => setError(""));

  async function start() {
    setError("");
    setReply(null);
    setTiming(null);
    setRunning(true);
    try {
      await withCamera(async (grab) => {
        let t = newTiming();
        for (;;) {  // 정지를 누르면 grab() 이 CANCELLED 를 던져 빠져나온다
          const jpeg = await grab();
          const sent = performance.now();
          let r: RecognizeReply;
          try {
            r = await recognize(jpeg);
          } catch (e) {
            if (e instanceof ApiError && e.status === 503) { await wait(200); continue; }
            throw e;
          }
          t = addTiming(t, r, performance.now() - sent);
          setTiming(t);
          setReply(r);
        }
      });
    } catch (e) {
      if (!(e instanceof Error && e.message === CANCELLED)) setError(describe(e));
    } finally {
      setRunning(false);
    }
  }

  const face = reply?.status === "ok";
  const tone = !reply || !face ? "" : reply.match ? "ok" : "fail";
  return (
    <Page page="test" eyebrow="얼굴 인식 테스트" health={health}>
      <Stage video={video} on={running} />

      <section className="panel">
        {running
          ? <button type="button" className="ghost" onClick={() => { cancelled.current = true; }}>정지</button>
          : <button type="button" className="big" disabled={!health?.ok} onClick={start}>인식 시작</button>}
        {!running && (
          <p className="note">
            카메라 프레임마다 등록된 모든 얼굴과의 유사도(코사인)를 보여 줍니다. 문턱 이상이면 일치.
            등록은 "등록 (관리자)" 화면에서 사진 한 장으로 합니다. 등록 없이 두 얼굴만 비교하려면 "사진 비교".
          </p>
        )}
        {error && <p className="admin-error" role="alert">{error}</p>}
      </section>

      {reply && (
        <section className={`panel result ${tone}`} aria-live="polite">
          <h2>{face ? (reply.match ? `일치 · ${reply.match}` : "일치하는 사람 없음") : (HINT[reply.status] ?? reply.status)}</h2>
          {reply.scores && reply.scores.length === 0 && <p>등록된 얼굴이 없습니다 — 등록 화면에서 먼저 등록하세요.</p>}
          {reply.scores && reply.scores.length > 0 && (
            <table className="scores">
              <tbody>
                {reply.scores.map((s) => (
                  <tr key={s.user_id} className={s.score >= reply.threshold ? "hit" : ""}>
                    <td>{s.user_id}</td>
                    <td><div className="bar"><div style={{ width: `${Math.min(100, Math.max(0, s.score) * 100)}%` }} /></div></td>
                    <td className="num">{s.score.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="meta">문턱 {reply.threshold}{reply.face && ` · ${faceText(reply.face)}`}</p>
          {timing && <TimingLine t={timing} />}
        </section>
      )}
    </Page>
  );
}
