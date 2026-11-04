param(
    [Parameter(Mandatory=$true)][string]$OptionFile,
    [Parameter(Mandatory=$true)][string]$SpotFile,
    [Parameter(Mandatory=$true)][string]$MarketPython,
    [Parameter(Mandatory=$true)][string]$HedgePython,
    [Parameter(Mandatory=$true)][string]$Upstream,
    [Parameter(Mandatory=$true)][string]$DataDir,
    [Parameter(Mandatory=$true)][string]$ResultDir,
    [switch]$PrepareOnly
)
$ErrorActionPreference='Stop'
$revision=git -C $Upstream rev-parse HEAD
if($LASTEXITCODE -ne 0 -or $revision -ne '6fa18790f14b56c99407ef987406459fa6e3b988'){
    throw 'Use the pinned neuralSDE-marketmodel revision.'
}
New-Item -ItemType Directory -Force -Path $DataDir,$ResultDir | Out-Null
$config=Join-Path $PSScriptRoot 'experiment.json'
$fingerprint=(@($OptionFile,$SpotFile,$config) | Get-FileHash | ForEach-Object Hash) -join ' '
$manifest=Join-Path $DataDir 'inputs.sha256'
if((Test-Path $manifest) -and ((Get-Content $manifest -Raw).Trim() -ne $fingerprint)){
    throw 'Inputs or configuration changed. Use a new data directory.'
}
$fingerprint | Set-Content -Encoding ascii $manifest
function Stage($Name,$Exe,$Script,$Arguments,$Outputs) {
    $done=Join-Path $DataDir ".$Name.done"
    if((Test-Path $done) -and -not ($Outputs | Where-Object {-not (Test-Path -LiteralPath $_)})){
        Write-Output "Resume: $Name complete"
        return
    }
    & $Exe (Join-Path $PSScriptRoot $Script) @Arguments
    if($LASTEXITCODE -ne 0){throw "$Script failed ($LASTEXITCODE)"}
    foreach($file in $Outputs){if(-not (Test-Path -LiteralPath $file)){throw "Missing output $file"}}
    New-Item -ItemType File -Force -Path $done | Out-Null
}
Stage select $MarketPython 'select_episodes.py' @($OptionFile,$SpotFile,"$DataDir/episodes.csv",$config) @("$DataDir/episodes.csv")
Stage ledger $MarketPython 'contract_ledger.py' @($OptionFile,$SpotFile,"$DataDir/episodes.csv","$DataDir/episode_days.csv") @("$DataDir/episode_days.csv")
Stage bs $MarketPython 'bs_baseline.py' @("$DataDir/episode_days.csv","$DataDir/bs_ledger.csv",$config) @("$DataDir/bs_ledger.csv")
Stage surfaces $MarketPython 'market_surfaces.py' @($OptionFile,$SpotFile,$Upstream,"$DataDir/surfaces.npz") @("$DataDir/surfaces.npz")
Stage factors $MarketPython 'market_factors.py' @("$DataDir/surfaces.npz",$Upstream,"$DataDir/factors.npz",$config) @("$DataDir/factors.npz")
Stage market $MarketPython 'joint_market.py' @("$DataDir/factors.npz",$SpotFile,$Upstream,$DataDir,$config) @("$DataDir/joint_paths.npz","$DataDir/boundary_diagnostics.csv","$DataDir/joint.weights.h5")
if($PrepareOnly){return}
Stage hedge $HedgePython 'hedge_experiment.py' @($DataDir,$ResultDir,$config) @("$DataDir/dh_ledger.csv","$ResultDir/training.csv")
Stage report $MarketPython 'report.py' @($DataDir,$ResultDir,$config) @("$ResultDir/summary.csv","$ResultDir/comparison_5bp.png")
