# SilverGuard v2.0 Deployment Package Builder
# Run: powershell -ExecutionPolicy Bypass -File build_package.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "       SilverGuard v2.0 - Building Deployment Package"
Write-Host "============================================================"
Write-Host ""

$version = "v2.0"
$releaseDir = "SilverGuard_$version"
$zipName = "SilverGuard_$version.zip"

# Clean temp folder
if (Test-Path $releaseDir) {
    Remove-Item $releaseDir -Recurse -Force
}
New-Item -ItemType Directory -Path $releaseDir | Out-Null

Write-Host "[1/4] Copying core files..."

# Core Python files
$coreFiles = @(
    "main.py",
    "dashboard.py",
    "utils.py",
    "stgcn.py",
    "voice_module.py",
    "skeleton_avatar.py",
    "offline_mode.py",
    "download_model.py",
    "requirements.txt"
)

foreach ($file in $coreFiles) {
    if (Test-Path $file) {
        Copy-Item $file $releaseDir\
        Write-Host "   + $file"
    }
}

# core folder
if (Test-Path "core") {
    Copy-Item "core" "$releaseDir\core" -Recurse
    Write-Host "   + core\"
}

# models folder
if (Test-Path "models") {
    New-Item -ItemType Directory -Path "$releaseDir\models" | Out-Null
    Copy-Item "models\*.pt" "$releaseDir\models\" -ErrorAction SilentlyContinue
    Copy-Item "models\*.pth" "$releaseDir\models\" -ErrorAction SilentlyContinue
    Write-Host "   + models\"
}

Write-Host ""
Write-Host "[2/4] Copying avatar images..."

# Avatar part images
$avatarImages = @(
    "head.png", "torso.png",
    "l_arm_up.png", "l_arm_low.png", "r_arm_up.png", "r_arm_low.png",
    "l_leg_up.png", "l_leg_low.png", "r_leg_up.png", "r_leg_low.png",
    "background.png"
)

foreach ($img in $avatarImages) {
    if (Test-Path $img) {
        Copy-Item $img $releaseDir\
    }
}
Write-Host "   + 11 avatar images"

Write-Host ""
Write-Host "[3/4] Copying scripts and docs..."

# Batch files
$batFiles = @(
    "INSTALL.bat",
    "Start_SilverGuard.bat"
)

foreach ($bat in $batFiles) {
    if (Test-Path $bat) {
        Copy-Item $bat $releaseDir\
        Write-Host "   + $bat"
    }
}

# Doc files
$docFiles = @(
    "QUICK_START.md",
    "FEATURE_SPEC.md",
    "Readme.md"
)

foreach ($doc in $docFiles) {
    if (Test-Path $doc) {
        Copy-Item $doc $releaseDir\
        Write-Host "   + $doc"
    }
}

# Create empty data folder structure
Write-Host ""
Write-Host "[4/4] Creating data folder structure..."

New-Item -ItemType Directory -Path "$releaseDir\data" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data\alert_images" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data\verified_falls" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data\false_alarms" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data\videos" -Force | Out-Null

# Create .gitkeep files
"" | Out-File "$releaseDir\data\alert_images\.gitkeep"
"" | Out-File "$releaseDir\data\verified_falls\.gitkeep"
"" | Out-File "$releaseDir\data\false_alarms\.gitkeep"
"" | Out-File "$releaseDir\data\videos\.gitkeep"

Write-Host "   + data\ folder structure"

# Create ZIP
Write-Host ""
Write-Host "Creating ZIP file..."

if (Test-Path $zipName) {
    Remove-Item $zipName -Force
}

Compress-Archive -Path $releaseDir -DestinationPath $zipName -CompressionLevel Optimal

# Show result
$zipSize = (Get-Item $zipName).Length / 1MB

Write-Host ""
Write-Host "============================================================"
Write-Host "       BUILD COMPLETE!"
Write-Host "============================================================"
Write-Host ""
Write-Host "   File: $zipName"
Write-Host "   Size: $([math]::Round($zipSize, 2)) MB"
Write-Host ""
Write-Host "   Distribution:"
Write-Host "   1. Share $zipName"
Write-Host "   2. Extract ZIP"
Write-Host "   3. Run INSTALL.bat (as Admin)"
Write-Host "   4. Run Start_SilverGuard.bat"
Write-Host ""
Write-Host "============================================================"
