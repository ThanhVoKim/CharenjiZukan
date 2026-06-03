// Two-triangle affine warp cho mouth sprite, port nguyên từ
// references/canvas-overlay-pattern.md (Drawing Helpers). Tham số hoá theo
// MouthTrack để dùng được cho nhiều asset (không hardcode mouth_track tĩnh).
import type {MouthPoint, MouthState, MouthTrack} from './manifest';

export const applyMouthCalibration = (
  track: MouthTrack,
  quad: MouthPoint[],
): MouthPoint[] => {
  const calibration = track.calibration ?? {offset: [0, 0], scale: 1, rotation: 0};
  if (!track.calibrationApplied) {
    return quad.map(([x, y]) => [x, y] as MouthPoint);
  }

  const offset = calibration.offset ?? [0, 0];
  const scale = calibration.scale ?? 1;
  const rotation = ((calibration.rotation ?? 0) * Math.PI) / 180;
  const centerX = quad.reduce((sum, [x]) => sum + x, 0) / quad.length;
  const centerY = quad.reduce((sum, [, y]) => sum + y, 0) / quad.length;
  const cos = Math.cos(rotation);
  const sin = Math.sin(rotation);

  return quad.map(([x, y]) => {
    const dx = (x - centerX) * scale;
    const dy = (y - centerY) * scale;
    return [
      dx * cos - dy * sin + centerX + offset[0],
      dx * sin + dy * cos + centerY + offset[1],
    ] as MouthPoint;
  });
};

const computeAffine = (
  sourceA: MouthPoint,
  sourceB: MouthPoint,
  sourceC: MouthPoint,
  targetA: MouthPoint,
  targetB: MouthPoint,
  targetC: MouthPoint,
) => {
  const [sx0, sy0] = sourceA;
  const [sx1, sy1] = sourceB;
  const [sx2, sy2] = sourceC;
  const [dx0, dy0] = targetA;
  const [dx1, dy1] = targetB;
  const [dx2, dy2] = targetC;
  const denominator = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);

  if (denominator === 0) return null;

  return {
    a: (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denominator,
    b: (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denominator,
    c: (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denominator,
    d: (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denominator,
    e:
      (dx0 * (sx1 * sy2 - sx2 * sy1) +
        dx1 * (sx2 * sy0 - sx0 * sy2) +
        dx2 * (sx0 * sy1 - sx1 * sy0)) /
      denominator,
    f:
      (dy0 * (sx1 * sy2 - sx2 * sy1) +
        dy1 * (sx2 * sy0 - sx0 * sy2) +
        dy2 * (sx0 * sy1 - sx1 * sy0)) /
      denominator,
  };
};

const drawTriangle = (
  context: CanvasRenderingContext2D,
  image: CanvasImageSource,
  sourceA: MouthPoint,
  sourceB: MouthPoint,
  sourceC: MouthPoint,
  targetA: MouthPoint,
  targetB: MouthPoint,
  targetC: MouthPoint,
) => {
  const matrix = computeAffine(sourceA, sourceB, sourceC, targetA, targetB, targetC);
  if (!matrix) return;

  context.save();
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.beginPath();
  context.moveTo(targetA[0], targetA[1]);
  context.lineTo(targetB[0], targetB[1]);
  context.lineTo(targetC[0], targetC[1]);
  context.closePath();
  context.clip();
  context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
  context.drawImage(image, 0, 0);
  context.restore();
};

const drawWarpedSprite = (
  context: CanvasRenderingContext2D,
  sprite: HTMLImageElement,
  quad: MouthPoint[],
) => {
  const spriteWidth = sprite.naturalWidth || sprite.width;
  const spriteHeight = sprite.naturalHeight || sprite.height;
  if (!spriteWidth || !spriteHeight) return;

  const s0: MouthPoint = [0, 0];
  const s1: MouthPoint = [spriteWidth, 0];
  const s2: MouthPoint = [spriteWidth, spriteHeight];
  const s3: MouthPoint = [0, spriteHeight];
  const [q0, q1, q2, q3] = quad;

  drawTriangle(context, sprite, s0, s1, s2, q0, q1, q2);
  drawTriangle(context, sprite, s0, s2, s3, q0, q2, q3);
};

export const drawMotionPngTuberMouth = (
  canvas: HTMLCanvasElement,
  sprites: Partial<Record<MouthState, HTMLImageElement>>,
  mouthState: MouthState,
  trackFrameIndex: number,
  track: MouthTrack,
) => {
  const context = canvas.getContext('2d');
  if (!context) return;

  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.imageSmoothingEnabled = true;

  const frame = track.frames[trackFrameIndex];
  if (!frame?.valid) return;

  const sprite = sprites[mouthState] ?? sprites.open ?? sprites.closed;
  if (!sprite) return;

  drawWarpedSprite(context, sprite, applyMouthCalibration(track, frame.quad));
};
