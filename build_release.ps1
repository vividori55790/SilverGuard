# Build Release Package for SilverGuard
$releaseDir = "SilverGuard_v2.0_Release"

# 1. Clean previous release
if (Test-Path $releaseDir) {
    Remove-Item -Path $releaseDir -Recurse -Force
}
New-Item -ItemType Directory -Path $releaseDir | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data" | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data\alert_images" | Out-Null

# 2. Copy Code & Assets (Root Files)
$includes = @(
    "main.py", "dashboard.py", "utils.py", "stgcn.py", 
    "voice_module.py", "skeleton_avatar.py", "offline_mode.py",
    "requirements.txt", "setup_app.bat", "run_app.bat", "Readme.md",
    "*.png" # Copy all asset images
)

foreach ($pattern in $includes) {
    Copy-Item -Path $pattern -Destination $releaseDir -Force -ErrorAction SilentlyContinue
}

# 3. Copy Directories
Copy-Item -Path "core" -Destination $releaseDir -Recurse
Copy-Item -Path "models" -Destination $releaseDir -Recurse

# 4. Copy Data (Selective)
# Only copy the hospital DB and settings. Exclude raw training data.
Copy-Item -Path "data\건강_병원.csv" -Destination "$releaseDir\data" -ErrorAction SilentlyContinue
if (Test-Path "data\settings.json") { Copy-Item -Path "data\settings.json" -Destination "$releaseDir\data" }

# 5. Create Zip (Optional, requires .NET 4.5+ installed usually present)
$zipFile = "SilverGuard_v2.0_Setup.zip"
if (Test-Path $zipFile) { Remove-Item $zipFile }

Write-Host "✅ Release files collected in '$releaseDir'"
Write-Host "📦 To create a ZIP file, you can right-click the folder and select 'Send to -> Compressed (zipped) folder'."

# Try to zip using PowerShell built-in (if available in newer Windows)
try {
    Compress-Archive -Path "$releaseDir\*" -DestinationPath $zipFile -Force
    Write-Host "🎉 ZIP archive created successfully: $zipFile"
} catch {
    Write-Host "⚠️ Could not auto-zip. Please manually zip the '$releaseDir' folder."
}

pause
