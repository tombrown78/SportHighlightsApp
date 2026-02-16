# Deployment Rules

## When to Rebuild Docker Containers

Rebuild is required when:
- New Python packages added to `requirements.txt`
- New service files added to `backend/app/services/`
- Changes to `Dockerfile` or `docker-compose.yml`
- Changes to environment variable handling in code

```powershell
docker compose build backend
docker compose up -d
```

## When to Run Database Migrations

Migration is required when:
- New columns added to existing tables
- New tables created
- Column types changed
- Indexes added/removed

### Migration Process

1. Update `backend/app/models/database.py` with new columns/tables
2. Update `backend/app/models/schemas.py` with corresponding Pydantic models
3. Create migration script in `scripts/migrate_vX.sql`
4. Document migration command in README

### Migration Commands

```powershell
# Single column addition
docker exec sports-postgres psql -U sports -d sports_highlights -c "ALTER TABLE table_name ADD COLUMN IF NOT EXISTS column_name TYPE;"

# Run migration script
docker exec -i sports-postgres psql -U sports -d sports_highlights < scripts/migrate_vX.sql

# Verify migration
docker exec sports-postgres psql -U sports -d sports_highlights -c "\d table_name"
```

## Current Schema Migrations

### Version 2.0 (Team Classification)
```sql
ALTER TABLE players ADD COLUMN IF NOT EXISTS team_color VARCHAR(7);
```

## Environment Variables

When adding new environment variables:
1. Add to `.env.example` with documentation
2. Add to `backend/app/core/config.py` Settings class
3. Document in README under Configuration section

## Checklist for Significant Changes

- [ ] Update `requirements.txt` if new Python packages
- [ ] Update database models if schema changes
- [ ] Create migration script if schema changes
- [ ] Update `.env.example` if new env vars
- [ ] Update README with:
  - [ ] New features description
  - [ ] Rebuild instructions
  - [ ] Migration commands
  - [ ] New configuration options
- [ ] Test with fresh install AND upgrade path
