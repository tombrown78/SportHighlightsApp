-- Database Migration Script for Sports Highlights App
-- Run this script if upgrading from a previous version

-- ============ Version 2.0 Migration ============
-- Adds team_color column for team classification feature

-- Add team_color column to players table (if not exists)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'players' AND column_name = 'team_color'
    ) THEN
        ALTER TABLE players ADD COLUMN team_color VARCHAR(7);
        RAISE NOTICE 'Added team_color column to players table';
    ELSE
        RAISE NOTICE 'team_color column already exists';
    END IF;
END $$;

-- ============ Verify Schema ============
-- Run these queries to verify the migration

-- Check players table structure
-- \d players

-- Check all tables exist
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';

-- Check row counts
-- SELECT 'videos' as table_name, COUNT(*) FROM videos
-- UNION ALL SELECT 'players', COUNT(*) FROM players
-- UNION ALL SELECT 'player_segments', COUNT(*) FROM player_segments
-- UNION ALL SELECT 'actions', COUNT(*) FROM actions
-- UNION ALL SELECT 'clips', COUNT(*) FROM clips;
