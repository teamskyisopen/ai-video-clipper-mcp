<#
.SYNOPSIS
    Automated FFmpeg setup script for AI Video Clipper on Windows.
.DESCRIPTION
    Downloads the official Gyan.dev FFmpeg essentials release (which includes libass
    for subtitle rendering), extracts binaries to ./bin/, and verifies libass support.
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
$BinDir = Join-Path $ProjectRoot "bin"
$FfmpegExe = Join-Path $BinDir "ffmpeg.exe"
$ZipPath = Join-Path $BinDir "ffmpeg-release.zip"
$TempExtract = Join-Path $BinDir "temp_extract"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     AI Video Clipper - Windows FFmpeg Setup Utility       " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check if FFmpeg already installed in ./bin with libass
if (Test-Path $FfmpegExe) {
    Write-Host "[*] Checking existing FFmpeg installation in ./bin..." -ForegroundColor Yellow
    $versionOutput = & $FfmpegExe -version 2>&1 | Out-String
    if ($versionOutput -match "libass") {
        Write-Host "[OK] FFmpeg with libass is already installed in ./bin/!" -ForegroundColor Green
        Write-Host "     $FfmpegExe" -ForegroundColor Gray
        exit 0
    } else {
        Write-Host "[!] Existing FFmpeg lacks libass support. Downloading compatible build..." -ForegroundColor Yellow
    }
}

# 2. Ensure ./bin directory exists
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}

# 3. Download official Gyan.dev essentials build
$DownloadUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
Write-Host '[1/4] Downloading FFmpeg Essentials release with libass...' -ForegroundColor Yellow
Write-Host "      Source: $DownloadUrl" -ForegroundColor Gray

try {
    # Use BITS or standard Invoke-WebRequest with TLS 1.2+
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath -UseBasicParsing
} catch {
    Write-Host "[X] Download failed: $_" -ForegroundColor Red
    exit 1
}

# 4. Extract archive
Write-Host "[2/4] Extracting binaries..." -ForegroundColor Yellow
if (Test-Path $TempExtract) {
    Remove-Item -Recurse -Force $TempExtract
}
Expand-Archive -Path $ZipPath -DestinationPath $TempExtract -Force

# 5. Locate and move executables
Write-Host "[3/4] Installing executables into ./bin/..." -ForegroundColor Yellow
$extractedBins = Get-ChildItem -Path $TempExtract -Recurse -Filter "*.exe"
foreach ($exe in $extractedBins) {
    Move-Item -Path $exe.FullName -Destination $BinDir -Force
}

# Clean up temporary archive and extraction directory
Remove-Item -Recurse -Force $TempExtract -ErrorAction SilentlyContinue
Remove-Item -Force $ZipPath -ErrorAction SilentlyContinue

# 6. Verify libass support
Write-Host "[4/4] Verifying FFmpeg and libass subtitle support..." -ForegroundColor Yellow
if (Test-Path $FfmpegExe) {
    $checkOutput = & $FfmpegExe -version 2>&1 | Out-String
    if ($checkOutput -match "libass") {
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "[OK] Setup Complete! FFmpeg with libass is ready." -ForegroundColor Green
        Write-Host "     Location: $FfmpegExe" -ForegroundColor Gray
        Write-Host "============================================================" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "[!] Warning: FFmpeg installed, but '--enable-libass' flag was not found." -ForegroundColor Yellow
    }
} else {
    Write-Host "[X] Error: ffmpeg.exe was not found in $BinDir" -ForegroundColor Red
    exit 1
}
