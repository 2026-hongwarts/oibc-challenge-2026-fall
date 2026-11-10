# run_tune_xgb_apply.sh 의 시드 2·3 재현 단계를 잠시 멈춘다(제출 파일 만드는 동안 컴퓨터 자원 양보). 다시 실행하면 이어서 함.
# 찾는 문자열을 쪼개 두어 이 PowerShell 자신의 명령줄과 겹치지 않게 함.
$a = 'run_tune_xgb' + '_apply'
$b = 'train_site' + '_model'
Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.CommandLine -and (
        ($_.Name -eq 'bash.exe' -and $_.CommandLine -like "*$a*") -or
        ($_.Name -eq 'python.exe' -and $_.CommandLine -like "*$b*" -and $_.CommandLine -like '*s2x2*'))
} | ForEach-Object {
    Write-Output ("멈춤 PID {0}: {1}" -f $_.ProcessId, $_.CommandLine.Substring(0, [Math]::Min(120, $_.CommandLine.Length)))
    Stop-Process -Id $_.ProcessId -Force
}
