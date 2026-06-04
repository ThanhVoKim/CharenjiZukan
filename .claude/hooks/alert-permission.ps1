# ============================================================
# Claude Code Hook: Permission Alert
# Plays alert sound + shows popup window for 3 seconds
# when Claude needs user permission (PermissionRequest event)
# ============================================================

# 1. Launch popup in background FIRST (non-blocking, shows while sound plays)
$popupScript = Join-Path $PSScriptRoot "show-permission-popup.ps1"
if (Test-Path $popupScript) {
    Start-Process powershell -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", """$popupScript"""
    ) -WindowStyle Hidden
}

# 2. Play confirmation sound synchronously (blocks ~1-2s until sound finishes)
$soundPath = "C:\Windows\Media\Windows Logon.wav"
if (Test-Path $soundPath) {
    $player = New-Object System.Media.SoundPlayer $soundPath
    $player.PlaySync()
}
