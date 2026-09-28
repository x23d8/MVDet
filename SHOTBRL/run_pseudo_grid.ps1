param(
    [Parameter(Mandatory = $true)]
    [string]$DataPath,

    [Parameter(Mandatory = $true)]
    [string]$PseudoDir,

    [ValidateSet("wildtrack", "multiviewx")]
    [string]$Dataset = "wildtrack",

    [ValidateSet(0, 20, 45, 60)]
    [int]$DropRatio = 45,

    [double[]]$Lambdas = @(0.025, 0.05, 0.1, 0.2, 0.4),

    [int[]]$Seeds = @(1),

    [int]$Epochs = 10
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = (Resolve-Path -LiteralPath (Join-Path $scriptDir "..\MVDetBRL")).Path
$resolvedDataPath = (Resolve-Path -LiteralPath $DataPath).Path
$resolvedPseudoDir = (Resolve-Path -LiteralPath $PseudoDir).Path

Push-Location $projectDir
try {
    foreach ($mode in @("pseudo_only", "pseudo_confuse")) {
        foreach ($lambda in $Lambdas) {
            foreach ($seed in $Seeds) {
                $runName = "${mode}_lp${lambda}_seed${seed}"
                $arguments = @(
                    "main.py",
                    "-d", $Dataset,
                    "--variant", "default",
                    "--data_path", $resolvedDataPath,
                    "--loss", "brl",
                    "--drop_ratio", $DropRatio,
                    "--pseudo_mode", $mode,
                    "--pseudo_dir", $resolvedPseudoDir,
                    "--lambda_pseudo", $lambda,
                    "--brl_no_mirror",
                    "--epochs", $Epochs,
                    "--seed", $seed,
                    "--loginfo", "mvdet_$runName"
                )
                Write-Host "Running $runName"
                & python @arguments
                if ($LASTEXITCODE -ne 0) {
                    throw "Training failed for $runName (exit code $LASTEXITCODE)"
                }
            }
        }
    }
}
finally {
    Pop-Location
}
