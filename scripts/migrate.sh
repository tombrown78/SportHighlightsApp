#!/bin/bash
# Database Migration Script
# Run this if upgrading from a previous version

set -e

echo "=== Sports Highlights App - Database Migration ==="
echo ""

# Check if running in Docker context
if [ -f /.dockerenv ]; then
    # Running inside container
    PSQL_CMD="psql -U $POSTGRES_USER -d $POSTGRES_DB"
else
    # Running on host
    PSQL_CMD="docker exec sports-postgres psql -U sports -d sports_highlights"
fi

echo "Running migration..."
echo ""

# Add team_color column
$PSQL_CMD -c "ALTER TABLE players ADD COLUMN IF NOT EXISTS team_color VARCHAR(7);" 2>/dev/null || true

echo "Migration complete!"
echo ""

# Verify
echo "Verifying schema..."
$PSQL_CMD -c "\d players" | grep -E "team_color|Column"

echo ""
echo "=== Migration Successful ==="
