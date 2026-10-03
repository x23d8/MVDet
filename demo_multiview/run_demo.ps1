param(
    [string]$WildtrackPath = "F:\thesis_dataset\Wildtrack_dataset",
    [string]$MultiviewXPath = "F:\thesis_dataset\MultiviewX_dataset",
    [string]$WildtrackResults = "",
    [string]$MultiviewXResults = "",
    [string]$GaussianArchive = "C:\Users\ADMIN\Documents\mvdet_yolo26x_results.zip",
    [string]$ModelsConfig = "",
    [int]$Port = 8765
)

$localWildtrack = Join-Path $PSScriptRoot "..\downloads\wildtrack_smoke_repo\Wildtrack_dataset_full\Wildtrack_dataset"
if (-not (Test-Path -LiteralPath $WildtrackPath) -and (Test-Path -LiteralPath $localWildtrack)) {
    $WildtrackPath = $localWildtrack
}

$arguments = @((Join-Path $PSScriptRoot "server.py"), "--port", $Port)
if (Test-Path -LiteralPath $WildtrackPath) { $arguments += @("--wildtrack", $WildtrackPath) }
if (Test-Path -LiteralPath $MultiviewXPath) { $arguments += @("--multiviewx", $MultiviewXPath) }
if ($WildtrackResults) { $arguments += @("--wildtrack-results", $WildtrackResults) }
if ($MultiviewXResults) { $arguments += @("--multiviewx-results", $MultiviewXResults) }
if ($GaussianArchive) { $arguments += @("--gaussian-archive", $GaussianArchive) }
if ($ModelsConfig) { $arguments += @("--models-config", $ModelsConfig) }

Write-Host "MVDet Explorer: http://127.0.0.1:$Port"
& python @arguments
