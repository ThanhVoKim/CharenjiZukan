// Mouth state mode "cue" (V1): KHÔNG đọc audio, chỉ dựa vào segment.hasTts
// trong group_manifest. Trong segment có TTS, miệng chạy theo nhịp nói
// (cadence ~0.12–0.18s, theo SKILL.md) qua chuỗi closed/half/open; ngoài TTS
// (mute/gap/tail) miệng đóng. Amplitude/hybrid để V2.
import type {MouthState, Segment} from './manifest';

// Một chu kỳ "mở rồi khép" — trông như đang nói, không phải nhấp nháy theo track fps.
const TALK_CADENCE: MouthState[] = ['closed', 'half', 'open', 'half'];
const CADENCE_SECONDS = 0.15;

export const makeGetMouthState = (segments: Segment[], fps: number) => {
  return (globalFrame: number): MouthState => {
    const seg = segments.find(
      (s) => globalFrame >= s.startFrame && globalFrame < s.endFrame,
    );
    if (!seg || !seg.hasTts) return 'closed';

    const seconds = globalFrame / fps;
    const phase = Math.floor(seconds / CADENCE_SECONDS);
    return TALK_CADENCE[((phase % TALK_CADENCE.length) + TALK_CADENCE.length) % TALK_CADENCE.length];
  };
};
