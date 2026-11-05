param(
    [Parameter(Mandatory=$true)][string]$OptionFile,
    [Parameter(Mandatory=$true)][string]$SpotFile,
    [string]$MarketPython,
    [string]$HedgePython,
    [string]$Upstream,
    [string]$DataDir,
    [string]$OldData,
    [switch]$PrepareOnly,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$project = git -C $PSScriptRoot rev-parse --show-toplevel
if ($LASTEXITCODE) { throw 'Run from the study repository.' }
$work = Join-Path $project '工作区'
$support = Join-Path $work '运行支持'
if (!$MarketPython) { $MarketPython = Join-Path $support '环境/market/python.exe' }
if (!$HedgePython) { $HedgePython = Join-Path $support '环境/hedge313/Scripts/python.exe' }
if (!$Upstream) { $Upstream = Join-Path $support '临时文件/neuralSDE-marketmodel' }
if (!$DataDir) { $DataDir = Join-Path $work '数据/处理后/创新组合实验' }
if (!$OldData) { $OldData = Join-Path $work '数据/处理后/完整认购' }
$legacy = Join-Path $project '交付区/程序/完整认购'
$env:PYTHONPATH = (Resolve-Path $legacy).Path
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '2'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:PYTHONUTF8 = '1'
$config = Join-Path $PSScriptRoot 'study.json'
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$lock = [System.IO.File]::Open((Join-Path $DataDir '.study.lock'), 'OpenOrCreate', 'Write', 'None')
try {
    & $MarketPython -u (Join-Path $PSScriptRoot 'prepare.py') --options $OptionFile --spot $SpotFile --upstream $Upstream --legacy $legacy --config $config --out $DataDir --old-data $OldData
    if ($LASTEXITCODE) { throw "Market preparation failed ($LASTEXITCODE)" }
    if ($PrepareOnly) { return }
    & $HedgePython -u (Join-Path $PSScriptRoot 'verify.py')
    if ($LASTEXITCODE) { throw 'Numerical or ledger verification failed.' }
    & $HedgePython -u (Join-Path $PSScriptRoot 'train.py') $DataDir $config $OldData
    if ($LASTEXITCODE) { throw "Hedger training failed ($LASTEXITCODE)" }
    & $HedgePython -u (Join-Path $PSScriptRoot 'report.py') $DataDir $config
    if ($LASTEXITCODE) { throw "Evaluation failed ($LASTEXITCODE)" }
    $completion = Get-Content -Raw (Join-Path $DataDir 'evaluation/completion.json') | ConvertFrom-Json
    if ($completion.failures -or ($completion.evaluated + $completion.unavailable) -ne $completion.expected) {
        throw 'The evaluation is incomplete. See evaluation/failures.json.'
    }
    $publishFlag = @()
    if ($Publish) { $publishFlag += '--publish' }
    & $HedgePython -u (Join-Path $PSScriptRoot 'deliver.py') $DataDir $project @publishFlag
    if ($LASTEXITCODE) { throw 'Delivery failed. The verified evaluation remains available locally.' }
} finally {
    $lock.Dispose()
}


