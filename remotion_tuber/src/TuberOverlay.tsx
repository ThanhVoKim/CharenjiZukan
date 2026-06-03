import React from 'react';
import {AbsoluteFill, Img, staticFile, useVideoConfig} from 'remotion';
import type {CharacterConfig, MouthTrack, TuberInputProps} from './manifest';
import {MotionPngTuberCharacter} from './MotionPngTuberCharacter';
import {makeGetMouthState} from './mouthState';

// Tính ô character ra px. Giữ ĐÚNG aspect của mouth_track để body (frame
// 1920x1080) và canvas (nội tại 1920x1080) cùng scale như nhau, tránh lệch/méo.
// width ƯU TIÊN: hễ có width → suy height = width / aspect (bỏ qua height) nên
// thường chỉ cần đặt width. Chỉ khi không có width mới suy width từ height.
const resolveCharacterBox = (
  c: CharacterConfig,
  track: MouthTrack,
  compW: number,
  compH: number,
) => {
  const trackAspect = track.width / track.height;
  const wSrc =
    c.width ??
    (c.widthRatio !== undefined ? Math.round(c.widthRatio * compW) : undefined);
  const hSrc =
    c.height ??
    (c.heightRatio !== undefined ? Math.round(c.heightRatio * compH) : undefined);

  let width: number;
  let height: number;
  if (wSrc !== undefined) {
    // width ưu tiên → suy height giữ tỉ lệ (height/heightRatio bị bỏ qua)
    width = wSrc;
    height = Math.round(wSrc / trackAspect);
  } else if (hSrc !== undefined) {
    // chỉ có height → suy width
    height = hSrc;
    width = Math.round(hSrc * trackAspect);
  } else {
    // không cho gì → default theo heightRatio 0.6
    height = Math.round(0.6 * compH);
    width = Math.round(height * trackAspect);
  }
  const left = c.left ?? Math.round((c.leftRatio ?? 0.6) * compW);
  const top = c.top ?? Math.round((c.topRatio ?? 0.3) * compH);
  return {left, top, width, height};
};

export const TuberOverlay: React.FC<TuberInputProps> = ({
  manifest,
  mouthTrack,
  assetId,
}) => {
  const {fps, width, height} = useVideoConfig();

  // Thiếu dữ liệu → render khung trong suốt rỗng (không crash render driver).
  if (!manifest || !mouthTrack) {
    return <AbsoluteFill />;
  }

  const box = resolveCharacterBox(manifest.character ?? {}, mouthTrack, width, height);
  const clipInset = manifest.character?.clipInset;
  const sharedStyle: React.CSSProperties = {
    position: 'absolute',
    left: box.left,
    top: box.top,
    width: box.width,
    height: box.height,
    clipPath:
      clipInset && clipInset !== '0px' ? `inset(${clipInset})` : undefined,
  };

  const bodyBase = `pngtuber/${assetId}/body-transparent`;
  const spriteBase = `pngtuber/${assetId}/mouth`;
  const getMouthState = makeGetMouthState(manifest.segments ?? [], fps);

  const renderBody = (trackFrameIndex: number) => (
    <Img
      src={staticFile(
        `${bodyBase}/frame-${String(trackFrameIndex).padStart(3, '0')}.png`,
      )}
      style={{width: '100%', height: '100%', objectFit: 'fill', display: 'block'}}
    />
  );

  // AbsoluteFill mặc định nền trong suốt → output PNG giữ alpha.
  return (
    <AbsoluteFill>
      <MotionPngTuberCharacter
        compositionFps={fps}
        frameOffset={manifest.groupStartFrame ?? 0}
        mouthTrack={mouthTrack}
        spriteBasePath={spriteBase}
        renderBody={renderBody}
        getMouthState={getMouthState}
        bodyStyle={sharedStyle}
        canvasStyle={sharedStyle}
      />
    </AbsoluteFill>
  );
};
