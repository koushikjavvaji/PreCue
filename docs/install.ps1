# Installs (or updates) PreCue in your user folder and starts it. On Windows, paste into PowerShell:
#   irm https://getprecue.vercel.app/install.ps1 | iex
# Updating keeps your keys (.env), your installed libraries (.venv) and your reports (data\).
$ErrorActionPreference = "Stop"

$dir = Join-Path $HOME "PreCue"
$zip = "https://github.com/koushikjavvaji/PreCue/archive/refs/heads/main.zip"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("precue-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Write-Host "Downloading PreCue..."
    Invoke-WebRequest -Uri $zip -OutFile (Join-Path $tmp "precue.zip") -UseBasicParsing
    Expand-Archive -Path (Join-Path $tmp "precue.zip") -DestinationPath $tmp
    $source = Join-Path $tmp "PreCue-main"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    # Copy the program; never overwrite your keys, your libraries or your reports.
    Get-ChildItem -Path $source -Force | Where-Object { $_.Name -notin @(".env", ".venv", "data") } |
        Copy-Item -Destination $dir -Recurse -Force
} finally {
    Remove-Item -Recurse -Force $tmp
}

Write-Host "PreCue is in $dir. Next time, double-click PreCue.bat there. Starting it now..."
& (Join-Path $dir "PreCue.bat")
