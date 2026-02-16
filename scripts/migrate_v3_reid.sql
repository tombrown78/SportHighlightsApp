-- Database Migration Script for Sports Highlights App
-- Version 3.0: Appearance-Based Player Re-Identification
-- Run this script if upgrading from a previous version

-- ============ Version 3.0 Migration ============
-- Adds appearance-based re-identification columns to players table

-- Add appearance_embedding column (stores 512-dim float32 array as binary)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'players' AND column_name = 'appearance_embedding'
    ) THEN
        ALTER TABLE players ADD COLUMN appearance_embedding BYTEA;
        RAISE NOTICE 'Added appearance_embedding column to players table';
    ELSE
        RAISE NOTICE 'appearance_embedding column already exists';
    END IF;
END $$;

-- Add appearance_cluster_id column (groups similar-looking players)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'players' AND column_name = 'appearance_cluster_id'
    ) THEN
        ALTER TABLE players ADD COLUMN appearance_cluster_id INTEGER;
        RAISE NOTICE 'Added appearance_cluster_id column to players table';
    ELSE
        RAISE NOTICE 'appearance_cluster_id column already exists';
    END IF;
END $$;

-- Add appearance_features column (human-readable features like colors)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'players' AND column_name = 'appearance_features'
    ) THEN
        ALTER TABLE players ADD COLUMN appearance_features JSONB;
        RAISE NOTICE 'Added appearance_features column to players table';
    ELSE
        RAISE NOTICE 'appearance_features column already exists';
    END IF;
END $$;

-- Add merged_track_ids column (tracks that were merged into this player)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'players' AND column_name = 'merged_track_ids'
    ) THEN
        ALTER TABLE players ADD COLUMN merged_track_ids JSONB;
        RAISE NOTICE 'Added merged_track_ids column to players table';
    ELSE
        RAISE NOTICE 'merged_track_ids column already exists';
    END IF;
END $$;

-- Create index on appearance_cluster_id for efficient grouping queries
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes 
        WHERE tablename = 'players' AND indexname = 'idx_players_appearance_cluster'
    ) THEN
        CREATE INDEX idx_players_appearance_cluster ON players(appearance_cluster_id);
        RAISE NOTICE 'Created index on appearance_cluster_id';
    ELSE
        RAISE NOTICE 'Index on appearance_cluster_id already exists';
    END IF;
END $$;

-- ============ Verify Schema ============
-- Run these queries to verify the migration

-- Check players table structure
-- \d players

-- Check new columns exist
-- SELECT column_name, data_type 
-- FROM information_schema.columns 
-- WHERE table_name = 'players' 
-- AND column_name IN ('appearance_embedding', 'appearance_cluster_id', 'appearance_features', 'merged_track_ids');

-- Check players with appearance data
-- SELECT id, jersey_number, appearance_cluster_id, appearance_features 
-- FROM players 
-- WHERE appearance_embedding IS NOT NULL 
-- LIMIT 10;
