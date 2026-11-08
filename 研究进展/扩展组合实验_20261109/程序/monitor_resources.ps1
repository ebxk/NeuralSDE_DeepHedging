param(
    [Parameter(Mandatory=$true)][int]$RunnerId,
    [Parameter(Mandatory=$true)][string]$Destination
)
$ErrorActionPreference = 'Stop'
$runner = Get-Process -Id $RunnerId
$start = $runner.StartTime
while ($true) {
    $runner = Get-Process -Id $RunnerId -ErrorAction SilentlyContinue
    if (!$runner -or $runner.StartTime -ne $start) { break }
    $tree = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId)
    $ids = [Collections.Generic.HashSet[int]]::new()
    [void]$ids.Add($RunnerId)
    do {
        $count = $ids.Count
        foreach ($row in $tree) {
            if ($ids.Contains([int]$row.ParentProcessId)) {
                [void]$ids.Add([int]$row.ProcessId)
            }
        }
    } while ($ids.Count -gt $count)
    $processes = @(Get-Process -Id @($ids) -ErrorAction SilentlyContinue)
    [pscustomobject]@{
        Time = (Get-Date).ToString('o')
        ElapsedSeconds = [math]::Round(((Get-Date)-$start).TotalSeconds,1)
        Processes = $processes.Count
        AggregateWorkingSetMiB = [math]::Round((($processes | Measure-Object -Property WorkingSet64 -Sum).Sum)/1MB,1)
        CurrentTreeCpuSeconds = [math]::Round(($processes | Measure-Object -Property CPU -Sum).Sum,1)
    } | Export-Csv -LiteralPath $Destination -Append -NoTypeInformation -Encoding UTF8
    Start-Sleep -Seconds 30
}

