// B1 + B2: chuẩn bị asset cho Remotion.
//  - Copy mouth_track.json + mouth/*.png vào public/pngtuber/<assetId>/
//  - Body: chromakey green-screen → body-transparent/frame-NNN.png (giữ kích thước, fps gốc)
//  - Đặt asset trong public/ để staticFile() load được khi render headless (B2)
//
// Dùng:
//   npm run prepare-assets                                  # auto-dò màu nền (4 góc) rồi key
//   npm run prepare-assets -- --color 0x08A702 --similarity 0.12 --blend 0.10
//   npm run prepare-assets -- --no-key                      # không chromakey (asset đã có alpha)
//
// LƯU Ý: body source là nền MÀU ĐẶC (vd loop_mouthless_h264.mp4 nền green ~0x08A702 — không phải
// green chuẩn 0x00FF00). Mặc định auto-dò màu nền từ 4 góc frame đầu rồi chromakey → alpha.
// Truyền --color để ghi đè khi auto-dò sai (nền không đồng nhất, góc dính nhân vật...).
import {execFileSync} from 'node:child_process';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectDir = resolve(__dirname, '..'); // remotion_tuber
const repoRoot = resolve(projectDir, '..'); // d:\project\CharenjiZukan

const argv = process.argv.slice(2);
const flag = (name: string): boolean => argv.includes(`--${name}`);
const arg = (name: string, def: string): string => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : def;
};

const assetId = arg('asset-id', 'nike_loop_fix');
const sourceDir = arg('asset-dir', resolve(repoRoot, 'assets/pngtuber', assetId));
const outDir = resolve(projectDir, 'public/pngtuber', assetId);
const doKey = !flag('no-key'); // mặc định: chromakey (giả định nền màu đặc / green-screen)
const colorExplicit = argv.includes('--color'); // user tự chỉ định màu key?
let keyColor = arg('color', '0x00FF00');
const similarity = arg('similarity', '0.10');
const blend = arg('blend', '0.10');
const bodySource = arg('body', resolve(sourceDir, 'loop_mouthless_h264.mp4'));

const mkdirp = (p: string) => mkdirSync(p, {recursive: true});

// Lấy RGB trung bình của 1 vùng crop ở frame đầu (qua ffmpeg, scale 1:1 → 1 pixel).
const sampleRGB = (src: string, crop: string): [number, number, number] => {
  const buf = execFileSync(
    'ffmpeg',
    ['-v', 'error', '-i', src, '-vf', `crop=${crop},scale=1:1`,
     '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    {maxBuffer: 1 << 20},
  );
  return [buf[0] ?? 0, buf[1] ?? 0, buf[2] ?? 0];
};

// Dò màu nền màu-đặc: lấy 4 góc frame đầu (crop biểu thức iw/ih) → median từng kênh.
// Nhân vật ở giữa nên 4 góc = nền; median bỏ qua góc lỡ dính nhân vật.
const detectBackgroundColor = (src: string): string => {
  const corners = [
    '80:80:0:0', '80:80:iw-80:0', '80:80:0:ih-80', '80:80:iw-80:ih-80',
  ].map((c) => sampleRGB(src, c));
  const median = (ch: number) => {
    const v = corners.map((s) => s[ch]).sort((a, b) => a - b);
    return Math.round((v[1] + v[2]) / 2);
  };
  const [r, g, b] = [median(0), median(1), median(2)];
  return '0x' + [r, g, b].map((n) => n.toString(16).padStart(2, '0')).join('').toUpperCase();
};

console.log(`[prepare-assets] assetId=${assetId}`);
console.log(`[prepare-assets] source=${sourceDir}`);
console.log(`[prepare-assets] out=${outDir}`);

if (!existsSync(sourceDir)) {
  console.error(`Không thấy asset dir: ${sourceDir}`);
  process.exit(1);
}

const trackPath = resolve(sourceDir, 'mouth_track.json');
const track = JSON.parse(readFileSync(trackPath, 'utf8')) as {frames: unknown[]};
const trackFrames = track.frames.length;

// Dirs
mkdirp(outDir);
mkdirp(join(outDir, 'mouth'));
const bodyDir = join(outDir, 'body-transparent');
if (existsSync(bodyDir)) rmSync(bodyDir, {recursive: true, force: true});
mkdirp(bodyDir);

// Copy mouth_track.json + sprites
copyFileSync(trackPath, join(outDir, 'mouth_track.json'));
for (const s of ['closed', 'half', 'open']) {
  const src = resolve(sourceDir, 'mouth', `${s}.png`);
  if (!existsSync(src)) {
    console.error(`Thiếu mouth sprite: ${src}`);
    process.exit(1);
  }
  copyFileSync(src, join(outDir, 'mouth', `${s}.png`));
}

// Body → transparent PNG sequence
if (doKey && !colorExplicit) {
  keyColor = detectBackgroundColor(bodySource);
  console.log(
    `[prepare-assets] auto-detect màu nền từ 4 góc frame đầu → ${keyColor} ` +
      `(truyền --color 0xRRGGBB để ghi đè).`,
  );
}
const vf = doKey
  ? `chromakey=${keyColor}:${similarity}:${blend},format=rgba`
  : `format=rgba`;
console.log(`[prepare-assets] ffmpeg extract body (vf="${vf}") ...`);
execFileSync(
  'ffmpeg',
  [
    '-y',
    '-i',
    bodySource,
    '-vf',
    vf,
    '-vsync',
    '0',
    '-start_number',
    '0',
    join(bodyDir, 'frame-%03d.png'),
  ],
  {stdio: 'inherit'},
);

const produced = readdirSync(bodyDir).filter((f) => f.endsWith('.png')).length;
console.log(
  `[prepare-assets] body-transparent: ${produced} frame (mouth_track: ${trackFrames} frame)`,
);
if (produced !== trackFrames) {
  console.warn(
    `[prepare-assets] CẢNH BÁO: số frame body (${produced}) khác mouth_track (${trackFrames}). ` +
      `body loop dùng modulo theo mouth_track.frames.length nên vẫn chạy, nhưng nên kiểm tra nguồn.`,
  );
}
if (doKey) {
  console.log(
    `[prepare-assets] LƯU Ý: đã chromakey màu ${keyColor}. Nếu nền không đồng nhất hoặc viền còn ám màu, ` +
      `chỉnh --similarity/--blend hoặc truyền --color đúng màu nền.`,
  );
}
console.log('[prepare-assets] done.');
