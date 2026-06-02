# Start QMD Index WebUI Server
# Run this script from the webui directory, or adjust the path to server.py below.

$scriptPath = Join-Path $PSScriptRoot "server.py"
$port = if ($args[0]) { $args[0] } else { 8090 }

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "python"
$psi.Arguments = "`"$scriptPath`" --port $port"
$psi.WorkingDirectory = $PSScriptRoot
$psi.UseShellExecute = $true
$psi.CreateNoWindow = $false
$psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$p = [System.Diagnostics.Process]::Start($psi)
Write-Output "QMD Index server started on port $port (PID: $($p.Id))"
