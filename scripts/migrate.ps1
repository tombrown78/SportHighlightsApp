# Database Migration Script for Windows
# Run this if upgrading from a previous version

Write-Host "=== Sports Highlights App - Database Migration ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
$dockerRunning = docker ps 2>$null
if (-not $?) {
    Write-Host "Error: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if postgres container is running
$postgresRunning = docker ps --filter "name=sports-postgres" --format "{{.Names}}" 2>$null
if (-not $postgresRunning) {
    Write-Host "Error: PostgreSQL container is not running." -ForegroundColor Red
    Write-Host "Run 'docker compose up -d postgres' first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Running migration..." -ForegroundColor Yellow
Write-Host ""

# Add team_color column
$result = docker exec sports-postgres psql -U sports -d sports_highlights -c "ALTER TABLE players ADD COLUMN IF NOT EXISTS team_color VARCHAR(7);" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Migration complete!" -ForegroundColor Green
} else {
    Write-Host "Migration result: $result" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Verifying schema..." -ForegroundColor Yellow

# Show players table structure
docker exec sports-postgres psql -U sports -d sports_highlights -c "\d players"

Write-Host ""
Write-Host "=== Migration Successful ===" -ForegroundColor Green
