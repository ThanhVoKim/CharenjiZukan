import React from 'react';
import {useCurrentFrame} from 'remotion';
import type {MouthState, MouthTrack} from './manifest';
import {MotionPngTuberMouthCanvas} from './MotionPngTuberMouthCanvas';

// Body + mouth canvas đồng bộ qua trackFrameIndex. frameOffset = groupStartFrame
// để body loop / cadence miệng LIÊN TỤC giữa các group (không reset ở đầu mỗi group).
export const MotionPngTuberCharacter: React.FC<{
  compositionFps: number;
  frameOffset: number;
  mouthTrack: MouthTrack;
  spriteBasePath: string;
  renderBody: (trackFrameIndex: number) => React.ReactNode;
  getMouthState: (globalFrame: number) => MouthState;
  bodyStyle: React.CSSProperties;
  canvasStyle: React.CSSProperties;
}> = ({
  compositionFps,
  frameOffset,
  mouthTrack,
  spriteBasePath,
  renderBody,
  getMouthState,
  bodyStyle,
  canvasStyle,
}) => {
  const localFrame = useCurrentFrame();
  const globalFrame = localFrame + frameOffset;

  const loopFrames = Math.round(
    (mouthTrack.frames.length / mouthTrack.fps) * compositionFps,
  );
  const loopFrame = ((globalFrame % loopFrames) + loopFrames) % loopFrames;
  const trackFrameIndex =
    Math.floor((loopFrame / compositionFps) * mouthTrack.fps) %
    mouthTrack.frames.length;

  return (
    <div style={{position: 'absolute', inset: 0}}>
      <div style={bodyStyle}>{renderBody(trackFrameIndex)}</div>
      <MotionPngTuberMouthCanvas
        mouthTrack={mouthTrack}
        spriteBasePath={spriteBasePath}
        mouthState={getMouthState(globalFrame)}
        trackFrameIndex={trackFrameIndex}
        style={canvasStyle}
      />
    </div>
  );
};
