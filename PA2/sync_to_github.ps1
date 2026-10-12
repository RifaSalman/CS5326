# Sync this working folder into ..\CS5326\PA2 and push to GitHub.
$ErrorActionPreference = "Stop"
$pa2 = Split-Path -Parent $MyInvocation.MyCommand.Path
$cs = Join-Path (Split-Path -Parent $pa2) "CS5326"
$dest = Join-Path $cs "PA2"

New-Item -ItemType Directory -Force -Path $dest | Out-Null
robocopy $pa2 $dest /E /XD .git __pycache__ .pytest_cache outputs /XF *.pyc sync_to_github.ps1 /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

Set-Location $cs
git add -A
$status = git status --porcelain
if ($status) {
  git -c user.email="rifa99salman@gmail.com" -c user.name="rifa99" commit -m "Update PA2 from local workspace."
  git push origin main
  Write-Host "Pushed to https://github.com/RifaSalman/CS5326"
} else {
  Write-Host "No changes to push."
  git push origin main
}
