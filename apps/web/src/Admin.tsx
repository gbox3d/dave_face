// 관리자 페이지(/admin): 토큰, 얼굴 등록(사진 한 장 — 카메라를 보며 안내에 맞추고 촬영), 목록·삭제, 접속 QR 두 장과 인쇄.
import { type FormEvent, useEffect, useRef, useState } from "react";
import { ApiError, type EnrollReply, type RecognizeReply, type User, deleteUser, enroll, listUsers, recognize } from "./api";
import { CANCELLED, HINT, Page, Stage, describe, faceText, useCamera, useClearOnRestore, useHealth, wait } from "./flow";
import { Qr } from "./Qr";

const TOKEN_KEY = "dave-face-admin-token";
// 이 페이지를 연 주소 기준 — 터널(HTTPS) 주소로 열어야 QR 로 연 폰에서 카메라가 켜진다.
const DEMO_URL = new URL(".", document.baseURI).href;
const ADMIN_URL = new URL("admin", document.baseURI).href;

// 접속 QR 은 /admin#t=<토큰> 으로 연다. URL 조각은 서버·프록시 로그에 남지 않는다.
// 읽고 나면 주소에서 지우고 탭 저장소(sessionStorage)에만 둔다 — 탭을 닫으면 사라진다.
function initialToken(): string {
  const fromHash = new URLSearchParams(location.hash.slice(1)).get("t") ?? "";
  try {
    if (fromHash) {
      sessionStorage.setItem(TOKEN_KEY, fromHash);
      history.replaceState(null, "", location.pathname + location.search);
      return fromHash;
    }
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return fromHash;
  }
}

function rememberToken(token: string) {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch { /* 저장소가 막혀 있으면 메모리에만 */ }
}

// 얼굴이 보이면 촬영 버튼이 열린다. 정면으로 가까이 찍을수록 나중 유사도가 높다.
function guidance(live: RecognizeReply | null): { text: string; ok: boolean } {
  if (!live) return { text: "카메라를 여는 중…", ok: false };
  if (live.status !== "ok") return { text: HINT[live.status] ?? live.status, ok: false };
  return { text: "얼굴 인식됨 — 촬영 버튼을 누르세요", ok: true };
}

export function Admin() {
  const health = useHealth();
  const [token, setToken] = useState(initialToken);
  const [enrollId, setEnrollId] = useState("");
  const [users, setUsers] = useState<User[] | null>(null);
  const [adminError, setAdminError] = useState("");
  const [running, setRunning] = useState(false);
  const [live, setLive] = useState<RecognizeReply | null>(null);
  const [done, setDone] = useState<EnrollReply | null>(null);
  const [enrollError, setEnrollError] = useState("");
  const wantCapture = useRef(false);
  const { video, cancelled, withCamera } = useCamera();
  const ready = health?.ok ?? false;
  useClearOnRestore(() => setEnrollError(""));

  useEffect(() => { rememberToken(token); }, [token]);
  useEffect(() => { if (token) void refreshUsers(); }, []); // QR 로 들어왔으면 목록부터 보여 준다

  // 목록·삭제 오류는 목록 패널 안에만 보인다.
  async function refreshUsers() {
    try { setUsers(await listUsers(token)); setAdminError(""); } catch (error) { setAdminError(describe(error)); }
  }

  async function enrollWithCamera(event: FormEvent) {
    event.preventDefault();
    const id = enrollId;
    setDone(null);
    setEnrollError("");
    setLive(null);
    wantCapture.current = false;
    setRunning(true);
    try {
      await withCamera(async (grab) => {
        await wait(600); // 자동 노출이 자리 잡을 시간
        for (;;) {
          const jpeg = await grab();
          if (wantCapture.current) {
            wantCapture.current = false;
            try {
              setDone(await enroll(token, id, jpeg));
              return; // 등록됐으면 카메라를 끈다
            } catch (e) {
              // 그 한 장만 거부된 것(얼굴 없음·정면 아님 등)이면 안내를 보여 주고 계속 본다.
              if (e instanceof ApiError && e.status === 422) { setEnrollError(describe(e)); continue; }
              throw e;
            }
          }
          try {
            setLive(await recognize(jpeg));
          } catch (e) {
            if (e instanceof ApiError && e.status === 503) { await wait(200); continue; }
            throw e;
          }
        }
      });
    } catch (e) {
      if (!(e instanceof Error && e.message === CANCELLED)) setEnrollError(describe(e));
    } finally {
      setRunning(false);
    }
    listUsers(token).then(setUsers).catch(() => {});
  }

  async function remove(id: string) {
    if (!window.confirm(`${id} 의 등록 사진 데이터를 지울까요?`)) return;
    try { await deleteUser(token, id); await refreshUsers(); } catch (error) { setAdminError(describe(error)); }
  }

  // 키보드가 넣는 공백·대문자를 바로 정리한다 — ID 는 영문 소문자·숫자·_·- 만 쓴다.
  const normalizeId = (value: string) => value.trim().toLowerCase();
  const idRule = "영문 소문자·숫자·_·- 1~32자";
  const adminQrUrl = token ? `${ADMIN_URL}#t=${encodeURIComponent(token)}` : "";
  const g = guidance(live);

  return (
    <Page page="admin" eyebrow="등록 (관리자)" health={health}>
      <Stage video={video} on={running} />

      {running && (
        <section className={`panel prompt ${g.ok ? "ready" : ""}`} aria-live="polite">
          <p className="meta">{enrollId} 등록 · 사진 한 장</p>
          <h2>{g.text}</h2>
          <p className="hint">{live?.face ? faceText(live.face) : " "}</p>
          {live?.match && live.scores && (
            <p className="note">이미 등록된 <strong>{live.match}</strong> 와 유사도 {live.scores[0].score.toFixed(3)}</p>
          )}
          {enrollError && <p className="admin-error" role="alert">{enrollError}</p>}
          <div className="row">
            <button type="button" disabled={!g.ok} onClick={() => { wantCapture.current = true; }}>촬영해 등록</button>
            <button type="button" className="ghost" onClick={() => { cancelled.current = true; }}>취소</button>
          </div>
        </section>
      )}

      {!running && done && (
        <section className="panel result ok" aria-live="polite">
          <h2>등록 완료 · {done.user_id}{done.replaced ? " (사진 교체)" : ""}</h2>
          <p>{faceText(done.face)}</p>
          <button type="button" onClick={() => setDone(null)}>확인</button>
        </section>
      )}
      {!running && !done && enrollError && <p className="admin-error" role="alert">{enrollError}</p>}

      <section className="panel">
        <label htmlFor="token">관리자 토큰</label>
        <input id="token" type="password" value={token} onChange={(e) => setToken(e.target.value.trim())} autoComplete="off" />
        <p className="note">관리자 QR 로 열면 자동으로 채워지고, 이 탭을 닫으면 지워집니다. Pi 에서는 <code>cat localfiles/admin-token</code>.</p>
      </section>

      <form className="panel" onSubmit={enrollWithCamera}>
        <label htmlFor="enroll-id">등록할 ID</label>
        <input id="enroll-id" value={enrollId} onChange={(e) => setEnrollId(normalizeId(e.target.value))}
               autoCapitalize="none" autoCorrect="off" spellCheck={false} required pattern="[a-z0-9_\-]{1,32}"
               title={idRule} placeholder="예: dave" disabled={running} />
        <button type="submit" disabled={!ready || running || !token}>카메라 켜서 등록</button>
        <p className="note">
          카메라에 얼굴이 잡히면 <strong>촬영해 등록</strong> 으로 사진 한 장을 등록합니다. 정면으로 가까이 찍을수록
          좋습니다. 같은 ID 는 사진이 바뀝니다.
        </p>
      </form>

      <section className="panel">
        <button type="button" className="ghost" disabled={!token || running} onClick={refreshUsers}>등록 목록</button>
        {adminError && <p className="admin-error" role="alert">{adminError}</p>}
        {users && (
          <ul className="users">
            {users.length === 0 && <li>등록된 사용자가 없습니다</li>}
            {users.map((u) => (
              <li key={u.user_id}>
                <span>
                  <strong>{u.user_id}</strong> · 눈 거리 {Math.round(u.eye_dist)}px · 선명도 {Math.round(u.blur)} ·
                  {" "}{new Date(u.created_at * 1000).toLocaleString()}
                </span>
                <button type="button" className="ghost danger" disabled={running} onClick={() => remove(u.user_id)}>삭제</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="panel qr-print">
        <h2 className="h">접속 QR</h2>
        <p className="note">
          지금 연 주소({new URL(DEMO_URL).host}) 기준입니다. 폰 카메라는 HTTPS 에서만 열리니 터널 주소로 연 뒤 만드세요.
          관리자 QR 에는 토큰이 들어 있습니다 — 아무에게나 주지 마세요.
        </p>
        <div className="qr-cards">
          <div className="qr-card">
            <h3>얼굴 인식 테스트</h3>
            <Qr url={DEMO_URL} label="테스트 접속 QR" />
            <span className="qr-url">{DEMO_URL}</span>
          </div>
          {adminQrUrl && (
            <div className="qr-card">
              <h3>관리자 (토큰 포함)</h3>
              <Qr url={adminQrUrl} label="관리자 접속 QR" />
              <span className="qr-url">{ADMIN_URL}#t=…</span>
            </div>
          )}
        </div>
        <button type="button" className="ghost" onClick={() => window.print()}>QR 인쇄</button>
      </section>
    </Page>
  );
}
