# Start QMD Index WebUI natively on Windows.
# Required instead of the Docker container because sqlite-vec vec0 is a Windows DLL
# (node_modules/sqlite-vec-windows-x64/vec0.dll) that the Linux container cannot load.
param([int]$Port = 8090)

$py = "C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"
$script = Join-Path $PSScriptRoot "server.py"
$log = Join-Path $PSScriptRoot "server_native.log"
$err = Join-Path $PSScriptRoot "server_native.err"

# free the port if something is already listening
Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep 1

Start-Process -FilePath $py -ArgumentList $script, "--host", "0.0.0.0", "--port", $Port `
    -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden

Write-Output "QMD Index WebUI (native, vec0) started on port $Port  (log: $log)"
