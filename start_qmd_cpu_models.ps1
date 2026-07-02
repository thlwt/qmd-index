# QMD CPU Model Server Launcher (Docker)
# Starts llama-server containers for embedding and reranker via Docker Compose

Write-Host "QMD Model Server Launcher (Docker)" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Stop local llama-server if running
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force

# Start Docker containers
Set-Location -LiteralPath "E:\QMD-Index"
docker compose up -d

Write-Host ""
Write-Host "Containers starting..." -ForegroundColor Yellow

# Wait and check
Start-Sleep -Seconds 3
docker compose ps

Write-Host ""
Write-Host "Logs:" -ForegroundColor Cyan
docker compose logs --tail=10

Write-Host ""
Write-Host "Use the following to test:" -ForegroundColor Yellow
Write-Host "  Embedding: curl http://127.0.0.1:2780/v1/embeddings -d '{\"model\":\"Qwen3-Embedding-0.6B-f16.gguf\",\"input\":\"test\"}'" -ForegroundColor Gray
Write-Host "  Reranker:  curl http://127.0.0.1:2781/v1/rerank -d '{\"model\":\"Qwen.Qwen3-Reranker-0.6B.Q8_0.gguf\",\"query\":\"test\",\"documents\":[\"doc1\"]}'" -ForegroundColor Gray
Write-Host ""
Write-Host "Manage with:" -ForegroundColor Yellow
Write-Host "  docker compose up -d       # Start" -ForegroundColor Gray
Write-Host "  docker compose down        # Stop" -ForegroundColor Gray
Write-Host "  docker compose logs -f     # Follow logs" -ForegroundColor Gray
Write-Host "  docker compose ps          # Status" -ForegroundColor Gray
