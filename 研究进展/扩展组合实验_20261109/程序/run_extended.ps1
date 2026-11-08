param(
    [Parameter(Mandatory=$true)][string]$OptionFile,
    [Parameter(Mandatory=$true)][string]$SpotFile,
    [int]$Workers = 6,
    [int]$InitialMarketPid = 0,
    [switch]$Publish,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$directory = Get-Item -LiteralPath $PSScriptRoot
while ($directory -and !(Test-Path -LiteralPath (Join-Path $directory.FullName '.git'))) {
    $directory = $directory.Parent
}
if (!$directory) { throw 'Run from the study repository.' }
$project = $directory.FullName
$work = Join-Path $project '工作区'
$envRoot = Join-Path $work '运行支持/环境'
$out = Join-Path $work '数据/处理后/扩展组合实验'
$marketPython = Join-Path $envRoot 'market/python.exe'
$hedgePython = Join-Path $envRoot 'hedge313/Scripts/python.exe'
$programs = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$(Join-Path $programs '创新组合实验');$(Join-Path $programs '完整认购')"
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:TF_NUM_INTEROP_THREADS = '2'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$lock = [IO.File]::Open((Join-Path $out '.expanded.lock'), 'OpenOrCreate', 'Write', 'None')
try {
    if (!$VerifyOnly) {
        if ($InitialMarketPid -and (Get-Process -Id $InitialMarketPid -ErrorAction SilentlyContinue)) {
            Wait-Process -Id $InitialMarketPid -ErrorAction SilentlyContinue
        }
        & $marketPython -u (Join-Path $PSScriptRoot 'expanded_market.py') --project $project --out $out --options $OptionFile --spot $SpotFile
        if ($LASTEXITCODE) { Write-Warning 'See R1/market_failures.json. Other branches will continue.' }
    }
    $flags = @()
    if ($Publish) { $flags += '--publish' }
    if ($VerifyOnly) { $flags += '--verify-only' }
    & $hedgePython -u (Join-Path $PSScriptRoot 'run_extended.py') --project $project --out $out --workers $Workers @flags
    if ($LASTEXITCODE) { throw "Expanded experiment exited with code $LASTEXITCODE. See the experiment logs." }
} finally {
    $lock.Dispose()
}

