param([switch]$Yes, [switch]$Vision)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $Yes) {
    $answer = Read-Host 'Install VSSS client dependencies in .venv? [Y/n]'
    if ($answer -and $answer -notmatch '^(y|yes)$') { return }
    if (-not $Vision) { $Vision = (Read-Host 'Also install the optional vision service? [y/N]') -match '^(y|yes)$' }
}
git submodule update --init --recursive
if ($LASTEXITCODE) { exit $LASTEXITCODE }
python -m venv .venv
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& .venv/Scripts/python.exe -m pip install -e .
if ($LASTEXITCODE) { exit $LASTEXITCODE }
if ($Vision) { & ./aruco_gps/setup.ps1 -Yes; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
