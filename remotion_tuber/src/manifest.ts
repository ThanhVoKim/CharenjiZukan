// Type cho group_manifest.json (Python sinh) + mouth_track.json (asset).
// File này KHÔNG import fs — an toàn để bundle vào composition (chạy trong Chromium).

export type MouthState = 'closed' | 'half' | 'open';
export type MouthPoint = [number, number];

export type MouthTrackFrame = {
  quad: MouthPoint[];
  valid: boolean;
};

export type MouthCalibration = {
  offset?: MouthPoint;
  scale?: number;
  rotation?: number;
};

export type MouthTrack = {
  fps: number;
  width: number;
  height: number;
  refSpriteSize?: MouthPoint;
  calibration?: MouthCalibration;
  calibrationApplied?: boolean;
  frames: MouthTrackFrame[];
};

export type BlockType = 'tts' | 'mute' | 'gap' | 'tail';

export type Segment = {
  segmentIndex: number;
  newStartMs?: number;
  newEndMs?: number;
  startFrame: number;
  endFrame: number;
  blockType: BlockType;
  hasTts: boolean;
};

// Vị trí + kích thước character. Ưu tiên px nếu có; nếu không, dùng ratio theo
// width/height của composition (B3 — tổng quát độ phân giải).
// KÍCH THƯỚC: width ưu tiên — hễ có width (hoặc widthRatio) thì height tự suy
// = width / aspect(mouth_track), bỏ qua height/heightRatio (giữ tỉ lệ, không méo).
// Chỉ khi không có width mới suy width từ height. Thường chỉ cần đặt width.
export type CharacterConfig = {
  left?: number;
  top?: number;
  width?: number;
  height?: number;
  leftRatio?: number;
  topRatio?: number;
  widthRatio?: number;
  heightRatio?: number;
  clipInset?: string;
};

export type MouthMode = 'cue' | 'amplitude' | 'hybrid';

export type GroupManifest = {
  schemaVersion?: number;
  groupId: string;
  groupIndex?: number;
  fps: number;
  width: number;
  height: number;
  groupStartFrame: number;
  groupEndFrame: number;
  renderStartFrame?: number;
  renderDurationFrames?: number;
  segments: Segment[];
  character: CharacterConfig;
  mouth?: {mode?: MouthMode};
  assetId?: string;
  // Đường dẫn nơi ghi overlay PNG (tuyệt đối). Nếu thiếu, driver tự đặt out/<groupId>.
  overlayDir?: string;
};

// inputProps mà render driver (Node) truyền vào composition.
// mouthTrack được driver đọc từ asset rồi nhúng vào đây để component dùng đồng bộ,
// tránh fetch bất đồng bộ trong calculateMetadata (chạy ở Chromium, không có fs).
export type TuberInputProps = {
  manifest: GroupManifest | null;
  mouthTrack: MouthTrack | null;
  assetId: string;
};
