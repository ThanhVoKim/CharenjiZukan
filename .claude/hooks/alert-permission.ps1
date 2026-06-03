# ============================================================
# Claude Code Hook: Permission Alert
# Plays alert sound + shows popup window for 3 seconds
# when Claude needs user confirmation (permission_prompt)
# ============================================================

# Play confirmation sound (immediate feedback)
$soundPath = "C:\Windows\Media\Ring04.wav"
if (Test-Path $soundPath) {
    $player = New-Object System.Media.SoundPlayer $soundPath
    $player.Play()
}

# Launch popup in a separate non-blocking process
$popupScript = Join-Path $PSScriptRoot "show-permission-popup.ps1"
if (Test-Path $popupScript) {
    Start-Process powershell -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", """$popupScript"""
    ) -WindowStyle Hidden
}
