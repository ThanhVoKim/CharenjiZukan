# ============================================================
# Claude Code Hook: Completion Alert
# Plays a distinct sound when Claude finishes responding (Stop event)
# ============================================================

# Play completion sound
$soundPath = "C:\Windows\Media\Alarm01.wav"
if (Test-Path $soundPath) {
    $player = New-Object System.Media.SoundPlayer $soundPath
    $player.Play()
}
