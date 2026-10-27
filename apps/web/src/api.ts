// 페이지가 놓인 곳 기준 상대 경로. /dave-face/ 에서 열리면 /dave-face/api, :3011/ 에서 열리면 /api — 빌드에 아무 값도 안 박는다.
export const API_BASE = new URL("api", document.baseURI).pathname.replace(/\/$/, "");
const TIMEOUT_MS = 8000;

export interface Health { ok: boolean; name: string; version: string; state: string; model: string }
export interface Face { bbox: number[]; eye_dist: number; yaw: number; roll: number; blur: number }
export interface RecognizeReply {
  status: string;                                   // ok | no_face
  threshold: number;                                // 이 이상이면 일치
  face?: Face;
  scores?: { user_id: string; score: number }[];    // 등록된 모두, 높은 점수부터
  match?: string | null;
  elapsed_ms: number;   // 서버가 요청을 받아 답하기까지 (JPEG 디코드 + NPU + 비교)
  npu_ms: number;       // 그중 NPU 호출(검출 + 임베딩)
}
export interface EmbedReply {
  status: string;        // ok | no_face
  threshold: number;
  face?: Face;
  embedding?: number[];  // 512-D, L2 정규화 — 두 임베딩의 내적이 코사인 유사도
  elapsed_ms: number;
  npu_ms: number;
}
export interface EnrollReply { user_id: string; replaced: boolean; face: Face }
export interface User { user_id: string; created_at: number; eye_dist: number; blur: number; yaw: number }

export class ApiError extends Error {
  constructor(public status: number, public body: Record<string, unknown> | null) {
    super(typeof body?.error === "string" ? body.error : `HTTP ${status}`);
  }
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store", signal: AbortSignal.timeout(TIMEOUT_MS), ...init });
  const body = r.status === 204 ? null : await r.json().catch(() => null);
  if (!r.ok) throw new ApiError(r.status, body);
  return body as T;
}

// 준비 중(503)에도 상태 본문을 돌려준다.
export async function fetchHealth(): Promise<Health> {
  const r = await fetch(`${API_BASE}/health`, { cache: "no-store", signal: AbortSignal.timeout(TIMEOUT_MS) });
  return (await r.json()) as Health;
}

// 프레임 한 장 → 얼굴 품질 + 등록된 모두와의 유사도.
export const recognize = (jpeg: Blob) =>
  call<RecognizeReply>("/recognize", { method: "POST", headers: { "content-type": "image/jpeg" }, body: jpeg });

// 프레임 한 장 → 임베딩 (저장하지 않는다). 사진 비교 화면이 쓴다.
export const embed = (jpeg: Blob) =>
  call<EmbedReply>("/embed", { method: "POST", headers: { "content-type": "image/jpeg" }, body: jpeg });

// 사진 한 장으로 등록 (같은 ID 는 교체).
export function enroll(token: string, userId: string, frame: Blob): Promise<EnrollReply> {
  const form = new FormData();
  form.append("user_id", userId);
  form.append("frame", frame, "frame.jpg");
  return call<EnrollReply>("/enroll", { method: "POST", headers: { "x-admin-token": token }, body: form });
}

export const listUsers = (token: string) => call<User[]>("/users", { headers: { "x-admin-token": token } });

export const deleteUser = (token: string, userId: string) =>
  call<null>(`/users/${encodeURIComponent(userId)}`, { method: "DELETE", headers: { "x-admin-token": token } });
