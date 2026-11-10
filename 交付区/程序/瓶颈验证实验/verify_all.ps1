param([string]$Project = (Get-Location).Path)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$studyHedge = Join-Path $Project '工作区/运行支持/环境/hedge313/Scripts/python.exe'
$studyMarket = Join-Path $Project '工作区/运行支持/环境/market/python.exe'
Push-Location (Join-Path $Project '工作区/当前研究/程序/瓶颈验证实验')
try {
    & $studyHedge -m unittest test_core test_account test_fold_inputs test_planner_experiment test_migration
    if ($LASTEXITCODE) { throw 'Hedge checks failed' }
    & $studyMarket -m unittest test_conditional_market
    if ($LASTEXITCODE) { throw 'Market checks failed' }
    & $studyHedge -c "import unittest,bootstrap,verify_policy; r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(verify_policy)); raise SystemExit(not r.wasSuccessful())"
    if ($LASTEXITCODE) { throw 'Inherited policy checks failed' }
    & $studyHedge verify_real_account.py
    if ($LASTEXITCODE) { throw 'Actual ledger acceptance failed' }
} finally {
    Pop-Location
}
