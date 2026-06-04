# ============================================================
# Claude Code Hook: Permission Popup Window
# Shows a centered, always-on-top popup that auto-closes
# after 3 seconds. Runs as a separate non-blocking process.
# ============================================================

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Build the popup form
$form = New-Object System.Windows.Forms.Form
$form.Text           = "Claude Code - Confirmation Required"
$form.Size           = New-Object System.Drawing.Size(460, 160)
$form.StartPosition  = "CenterScreen"
$form.TopMost        = $true
$form.FormBorderStyle = "FixedDialog"
$form.ControlBox     = $false
$form.BackColor      = [System.Drawing.Color]::FromArgb(255, 248, 225)  # light amber

# --- Title label ---
$title = New-Object System.Windows.Forms.Label
$title.Text      = "Claude is waiting for your confirmation!"
$title.AutoSize  = $true
$title.Font      = New-Object System.Drawing.Font("Segoe UI", 13, [System.Drawing.FontStyle]::Bold)
$title.ForeColor = [System.Drawing.Color]::FromArgb(180, 80, 0)
$title.Location  = New-Object System.Drawing.Point(25, 25)
$form.Controls.Add($title)

# --- Subtitle label ---
$sub = New-Object System.Windows.Forms.Label
$sub.Text      = "Please check the Claude Code terminal to approve."
$sub.AutoSize  = $true
$sub.Font      = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Regular)
$sub.ForeColor = [System.Drawing.Color]::FromArgb(80, 80, 80)
$sub.Location  = New-Object System.Drawing.Point(25, 65)
$form.Controls.Add($sub)

# --- Countdown label ---
$countdown = New-Object System.Windows.Forms.Label
$countdown.Text      = "Auto-closing in 3s..."
$countdown.AutoSize  = $true
$countdown.Font      = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Italic)
$countdown.ForeColor = [System.Drawing.Color]::Gray
$countdown.Location  = New-Object System.Drawing.Point(25, 95)
$form.Controls.Add($countdown)

# --- Timer: auto-close after 3 seconds ---
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 3000  # 3 seconds
$timer.Add_Tick({
    $timer.Stop()
    $timer.Dispose()
    $form.Close()
})
$timer.Start()

# --- Show modal (blocks only this process, not the main Claude session) ---
[void]$form.ShowDialog()
