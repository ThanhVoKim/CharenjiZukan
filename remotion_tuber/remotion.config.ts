// Chỉ dùng cho `remotion studio` / Remotion CLI khi debug bằng tay.
// Render vận hành chính (scripts/render-groups.ts) dùng API programmatic và
// truyền imageFormat trực tiếp vào renderFrames, KHÔNG đọc file này.
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('png'); // overlay cần alpha
Config.setOverwriteOutput(true);
