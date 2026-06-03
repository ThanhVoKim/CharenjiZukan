import React from 'react';
import {CalculateMetadataFunction, Composition} from 'remotion';
import {TuberOverlay} from './TuberOverlay';
import type {TuberInputProps} from './manifest';

// B3/B4: fps + độ phân giải + độ dài lấy ĐỘNG từ group_manifest (qua inputProps),
// không hardcode. calculateMetadata chạy ở Chromium nên chỉ đọc props (đã có sẵn
// manifest do driver nhúng vào inputProps), không dùng fs.
const calculateMetadata: CalculateMetadataFunction<TuberInputProps> = ({props}) => {
  const m = props.manifest;
  if (!m) return {};
  const duration =
    m.renderDurationFrames ?? m.groupEndFrame - m.groupStartFrame;
  return {
    fps: m.fps,
    width: m.width,
    height: m.height,
    durationInFrames: Math.max(1, Math.round(duration)),
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="TuberOverlay"
      component={TuberOverlay}
      // Placeholder — bị calculateMetadata override theo manifest thật.
      durationInFrames={1}
      fps={30}
      width={1920}
      height={1080}
      defaultProps={
        {manifest: null, mouthTrack: null, assetId: 'nike_loop_fix'} as TuberInputProps
      }
      calculateMetadata={calculateMetadata}
    />
  );
};
