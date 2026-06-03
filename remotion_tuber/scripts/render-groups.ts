// Render driver (Phase I): bundle Remotion MỘT LẦN, render overlay PNG alpha cho
// từng group. Python sẽ gọi script này với danh sách group_manifest.json.
//
// Dùng:
//   npm run render-groups -- <group_manifest.json> [more_manifests...]
//   npm run render-groups -- test-manifests/group_synthetic.json
//   options: --asset-id <id>  --out-dir <base>
//
// Mỗi group in ra một dòng kết quả có cấu trúc để Python parse:
//   __TUBER_RENDER_RESULT__={"groupId":"...","ok":true,"frames":90,...}
// Exit code != 0 nếu có group fail.
import {bundle} from '@remotion/bundler';
import {ensureBrowser, renderFrames, selectComposition} from '@remotion/renderer';
import {existsSync, mkdirSync, readdirSync, readFileSync} from 'node:fs';
import {dirname, isAbsolute, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectDir = resolve(__dirname, '..'); // remotion_tuber

const argv = process.argv.slice(2);
const arg = (name: string): string | undefined => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
};
const manifestArgs = argv.filter((a, i) => {
  if (a.startsWith('--')) return false;
  // bỏ giá trị đứng ngay sau một --flag
  if (i > 0 && argv[i - 1].startsWith('--')) return false;
  return true;
});

if (manifestArgs.length === 0) {
  console.error(
    'Usage: npm run render-groups -- <group_manifest.json> [more...] [--asset-id id] [--out-dir base]',
  );
  process.exit(2);
}

const assetIdArg = arg('asset-id');
const outBase = arg('out-dir')
  ? resolve(process.cwd(), arg('out-dir') as string)
  : resolve(projectDir, 'out');

const resolvePath = (p: string) => (isAbsolute(p) ? p : resolve(process.cwd(), p));

const main = async () => {
  await ensureBrowser();

  const entryPoint = resolve(projectDir, 'src/index.ts');
  console.log('[render-groups] bundling once...');
  // publicDir mặc định = projectDir/public → staticFile() resolve từ đó (B2).
  const serveUrl = await bundle({entryPoint});
  console.log('[render-groups] bundle ready.');

  let failures = 0;

  for (const mp of manifestArgs) {
    const manifestPath = resolvePath(mp);
    try {
      const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
      const assetId = manifest.assetId ?? assetIdArg ?? 'nike_loop_fix';

      const mouthTrackPath = resolve(
        projectDir,
        'public/pngtuber',
        assetId,
        'mouth_track.json',
      );
      if (!existsSync(mouthTrackPath)) {
        throw new Error(
          `Thiếu mouth_track trong public: ${mouthTrackPath}. Chạy "npm run prepare-assets" trước.`,
        );
      }
      const mouthTrack = JSON.parse(readFileSync(mouthTrackPath, 'utf8'));

      const inputProps = {manifest, mouthTrack, assetId};

      const composition = await selectComposition({
        serveUrl,
        id: 'TuberOverlay',
        inputProps,
      });

      const outputDir = manifest.overlayDir
        ? resolvePath(manifest.overlayDir)
        : join(outBase, manifest.groupId, 'overlay_frames');
      mkdirSync(outputDir, {recursive: true});

      await renderFrames({
        serveUrl,
        composition,
        inputProps,
        outputDir,
        imageFormat: 'png', // alpha
        imageSequencePattern: 'frame_[frame].[ext]',
        onStart: () =>
          console.log(
            `[${manifest.groupId}] render ${composition.durationInFrames} frame @ ` +
              `${composition.width}x${composition.height} ${composition.fps}fps`,
          ),
        onFrameUpdate: (f: number) => {
          if (f % 30 === 0) console.log(`[${manifest.groupId}] frame ${f}`);
        },
      });

      const produced = readdirSync(outputDir).filter((f) =>
        f.endsWith('.png'),
      ).length;
      const result = {
        groupId: manifest.groupId,
        ok: true,
        frames: composition.durationInFrames,
        produced,
        outputDir,
      };
      process.stdout.write(`__TUBER_RENDER_RESULT__=${JSON.stringify(result)}\n`);
    } catch (err) {
      failures++;
      const result = {
        groupId: manifestPath,
        ok: false,
        error: String((err as Error)?.stack ?? err),
      };
      process.stdout.write(`__TUBER_RENDER_RESULT__=${JSON.stringify(result)}\n`);
      console.error(err);
    }
  }

  process.exit(failures > 0 ? 1 : 0);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
