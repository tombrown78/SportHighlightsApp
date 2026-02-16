-- Initialize database schema for Sports Highlights App

-- Videos table
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    original_url VARCHAR(1024),
    file_path VARCHAR(512) NOT NULL,
    duration_seconds FLOAT,
    width INTEGER,
    height INTEGER,
    fps FLOAT,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Players detected in videos
CREATE TABLE IF NOT EXISTS players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    jersey_number VARCHAR(10),
    team VARCHAR(50),
    track_id INTEGER,
    confidence FLOAT,
    first_seen_frame INTEGER,
    last_seen_frame INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Player segments (time ranges where player is visible/active)
CREATE TABLE IF NOT EXISTS player_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id UUID NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Detected actions (shots, rebounds, etc.)
CREATE TABLE IF NOT EXISTS actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    player_id UUID REFERENCES players(id) ON DELETE SET NULL,
    action_type VARCHAR(50) NOT NULL,
    frame INTEGER NOT NULL,
    timestamp FLOAT NOT NULL,
    confidence FLOAT,
    action_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Generated clips
CREATE TABLE IF NOT EXISTS clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    player_id UUID REFERENCES players(id) ON DELETE SET NULL,
    title VARCHAR(255),
    file_path VARCHAR(512) NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration_seconds FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_players_video_id ON players(video_id);
CREATE INDEX IF NOT EXISTS idx_players_jersey_number ON players(jersey_number);
CREATE INDEX IF NOT EXISTS idx_player_segments_player_id ON player_segments(player_id);
CREATE INDEX IF NOT EXISTS idx_player_segments_video_id ON player_segments(video_id);
CREATE INDEX IF NOT EXISTS idx_actions_video_id ON actions(video_id);
CREATE INDEX IF NOT EXISTS idx_actions_player_id ON actions(player_id);
CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type);
CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id);
CREATE INDEX IF NOT EXISTS idx_clips_player_id ON clips(player_id);
