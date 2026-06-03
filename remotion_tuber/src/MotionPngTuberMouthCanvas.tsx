import React from 'react';
import {cancelRender, continueRender, delayRender, staticFile} from 'remotion';
import type {MouthState, MouthTrack} from './manifest';
import {drawMotionPngTuberMouth} from './mouthWarp';

const MOUTH_STATES: MouthState[] = ['closed', 'half', 'open'];

export const MotionPngTuberMouthCanvas: React.FC<{
  mouthTrack: MouthTrack;
  // base path tương đối trong public/, ví dụ "pngtuber/nike_loop_fix/mouth"
  spriteBasePath: string;
  mouthState: MouthState;
  trackFrameIndex: number;
  style: React.CSSProperties;
}> = ({mouthTrack, spriteBasePath, mouthState, trackFrameIndex, style}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const spritesRef = React.useRef<Partial<Record<MouthState, HTMLImageElement>>>({});
  const [ready, setReady] = React.useState(false);
  const [loadHandle] = React.useState(() =>
    delayRender('Load MotionPNGTuber mouth sprites'),
  );

  // Preload 3 sprite TRƯỚC khi render frame (skill rule). staticFile → same-origin,
  // không taint canvas nên drawImage/getImageData chạy được trong render headless.
  React.useEffect(() => {
    let canceled = false;

    Promise.all(
      MOUTH_STATES.map(
        (state) =>
          new Promise<[MouthState, HTMLImageElement]>((resolve, reject) => {
            const image = new Image();
            image.onload = () => resolve([state, image]);
            image.onerror = () =>
              reject(new Error(`Failed to load mouth sprite: ${state}`));
            image.src = staticFile(`${spriteBasePath}/${state}.png`);
          }),
      ),
    )
      .then((sprites) => {
        if (canceled) return;
        spritesRef.current = Object.fromEntries(sprites) as Partial<
          Record<MouthState, HTMLImageElement>
        >;
        setReady(true);
        continueRender(loadHandle);
      })
      .catch((error) => {
        if (!canceled) cancelRender(error);
      });

    return () => {
      canceled = true;
    };
  }, [loadHandle, spriteBasePath]);

  React.useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!ready || !canvas) return;
    drawMotionPngTuberMouth(
      canvas,
      spritesRef.current,
      mouthState,
      trackFrameIndex,
      mouthTrack,
    );
  }, [mouthState, ready, trackFrameIndex, mouthTrack]);

  // width/height nội tại = kích thước nguồn của mouth_track (vd 1920x1080);
  // style scale canvas xuống đúng ô character (giống body) để khớp tọa độ.
  return (
    <canvas
      ref={canvasRef}
      width={mouthTrack.width}
      height={mouthTrack.height}
      style={style}
    />
  );
};
