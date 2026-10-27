// 전면 카메라를 열고 프레임을 JPEG 로 뜬다. 미리보기만 CSS 로 거울처럼 보이고, 서버로 가는 픽셀은 뒤집지 않는다.
const LONG_SIDE = 1280;
const QUALITY = 0.85;

export async function openCamera(video: HTMLVideoElement): Promise<MediaStream> {
  // getUserMedia 는 보안 컨텍스트(HTTPS 또는 localhost)에서만 있다.
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error("insecure");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera(stream: MediaStream, video: HTMLVideoElement): void {
  stream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

export function captureJpeg(video: HTMLVideoElement, canvas: HTMLCanvasElement): Promise<Blob> {
  const scale = Math.min(1, LONG_SIDE / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("capture failed"))), "image/jpeg", QUALITY));
}
