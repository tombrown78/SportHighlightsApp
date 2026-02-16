'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams } from 'next/navigation';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Download,
  User,
  Activity,
  Clock,
  Loader2,
  AlertCircle,
  CheckCircle,
  Zap,
  Eye,
  Target,
  XCircle,
  RefreshCw,
  Circle,
  Hand,
  ArrowRight,
  Filter,
  List,
  Film,
  Trash2,
  Info,
  BarChart3,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Video {
  id: string;
  filename: string;
  status: string;
  duration_seconds: number;
  width: number;
  height: number;
  fps: number;
  player_count: number;
  action_count: number;
  error_message?: string;
}

interface Player {
  id: string;
  jersey_number: string;
  team: string;
  team_color?: string;  // RGB hex color
  confidence: number;
  segment_count: number;
  action_count: number;
}

interface Segment {
  id: string;
  start_time: number;
  end_time: number;
}

interface Action {
  id: string;
  action_type: string;
  timestamp: number;
  confidence: number;
  player_id?: string;
  player_jersey?: string;
  player_team?: string;
  frame?: number;
}

// All actions for the video (play-by-play)
interface VideoAction {
  id: string;
  video_id: string;
  action_type: string;
  timestamp: number;
  frame: number;
  confidence: number;
  player_id?: string;
  player_jersey?: string;
  player_team?: string;
  action_data?: Record<string, unknown>;
}

interface TimelineMarker {
  time: number;
  type: string;
  player_jersey?: string;
  action_type?: string;
  label?: string;
}

// Clip/Highlight interface
interface Clip {
  id: string;
  video_id: string;
  player_id?: string;
  title?: string;
  file_path: string;
  start_time: number;
  end_time: number;
  duration_seconds?: number;
  download_url?: string;
  created_at: string;
  player_jersey?: string;
}

interface ProcessingEvent {
  event: string;
  percent?: number;
  frame?: number;
  total_frames?: number;
  stage?: string;
  message?: string;
  track_id?: number;
  jersey_number?: string;
  confidence?: number;
  action_type?: string;
  player_track_id?: number;
  timestamp?: number;
  count?: number;
  class_name?: string;
  total_players?: number;
  players_count?: number;
  actions_count?: number;
}

interface LiveDetection {
  id: string;
  type: 'player' | 'action' | 'detection';
  message: string;
  timestamp: number;
  confidence?: number;
}

export default function AnalyzePage() {
  const params = useParams();
  const videoId = params.id as string;

  const videoRef = useRef<HTMLVideoElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const [video, setVideo] = useState<Video | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [actions, setActions] = useState<Action[]>([]);
  const [markers, setMarkers] = useState<TimelineMarker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [generatingClips, setGeneratingClips] = useState(false);
  
  // Processing state
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStage, setProcessingStage] = useState('');
  const [processingMessage, setProcessingMessage] = useState('');
  const [liveDetections, setLiveDetections] = useState<LiveDetection[]>([]);
  const [detectedPlayersCount, setDetectedPlayersCount] = useState(0);
  const [detectedActionsCount, setDetectedActionsCount] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showCancelConfirm, setShowCancelConfirm] = useState(false);

  // Play-by-play state
  const [allActions, setAllActions] = useState<VideoAction[]>([]);
  const [actionFilter, setActionFilter] = useState<string | null>(null);
  const [showPlayByPlay, setShowPlayByPlay] = useState(true);

  // Clips state
  const [clips, setClips] = useState<Clip[]>([]);
  const [deletingClipId, setDeletingClipId] = useState<string | null>(null);

  // Cancel video processing
  const cancelProcessing = async () => {
    if (!video) return;
    
    setIsCancelling(true);
    try {
      const response = await fetch(`${API_URL}/api/videos/${videoId}/cancel`, {
        method: 'POST',
      });
      
      if (response.ok) {
        setProcessingMessage('Cancelling...');
        setShowCancelConfirm(false);
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to cancel');
      }
    } catch (err) {
      console.error('Error cancelling:', err);
      alert('Failed to cancel processing');
    } finally {
      setIsCancelling(false);
    }
  };

  // Retry failed/cancelled video
  const retryProcessing = async () => {
    if (!video) return;
    
    try {
      const formData = new FormData();
      const response = await fetch(`${API_URL}/api/videos/${videoId}/retry`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.ok) {
        // Refresh video data
        fetchVideo();
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to retry');
      }
    } catch (err) {
      console.error('Error retrying:', err);
      alert('Failed to retry processing');
    }
  };

  // Connect to SSE for real-time progress
  useEffect(() => {
    if (!video || (video.status !== 'processing' && video.status !== 'queued')) {
      return;
    }

    const eventSource = new EventSource(`${API_URL}/api/videos/${videoId}/progress`);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data: ProcessingEvent = JSON.parse(event.data);
        
        switch (data.event) {
          case 'connected':
            setProcessingMessage('Connected to processing stream...');
            break;
            
          case 'progress':
            setProcessingProgress(data.percent || 0);
            if (data.stage) setProcessingStage(data.stage);
            break;
            
          case 'stage':
            setProcessingStage(data.stage || '');
            setProcessingMessage(data.message || '');
            break;
            
          case 'detection':
            setTotalDetections(data.count || 0);
            // Add to live feed (keep last 20)
            const detectionItem: LiveDetection = {
              id: `det-${Date.now()}-${Math.random()}`,
              type: 'detection',
              message: `Detected ${data.class_name} (Track #${data.track_id})`,
              timestamp: Date.now(),
              confidence: data.confidence
            };
            setLiveDetections(prev => [detectionItem, ...prev].slice(0, 20));
            break;
            
          case 'player':
            setDetectedPlayersCount(data.total_players || 0);
            const playerItem: LiveDetection = {
              id: `player-${Date.now()}-${Math.random()}`,
              type: 'player',
              message: `Player #${data.jersey_number} identified`,
              timestamp: Date.now(),
              confidence: data.confidence
            };
            setLiveDetections(prev => [playerItem, ...prev].slice(0, 20));
            break;
            
          case 'action':
            setDetectedActionsCount(data.count || 0);
            const actionItem: LiveDetection = {
              id: `action-${Date.now()}-${Math.random()}`,
              type: 'action',
              message: `${data.action_type?.replace('_', ' ')} at ${data.timestamp?.toFixed(1)}s`,
              timestamp: Date.now(),
              confidence: data.confidence
            };
            setLiveDetections(prev => [actionItem, ...prev].slice(0, 20));
            break;
            
          case 'complete':
            setProcessingProgress(100);
            setProcessingStage('complete');
            setProcessingMessage('Analysis complete!');
            // Refresh video data
            fetchVideo();
            eventSource.close();
            break;
          
          case 'cancelling':
            setProcessingMessage(data.message || 'Cancelling...');
            setProcessingStage('cancelling');
            break;
          
          case 'cancelled':
            setProcessingMessage('Processing cancelled');
            setProcessingStage('cancelled');
            // Refresh video data
            fetchVideo();
            eventSource.close();
            break;
            
          case 'error':
            setError(data.message || 'Processing failed');
            eventSource.close();
            break;
        }
      } catch (e) {
        console.error('Error parsing SSE event:', e);
      }
    };

    eventSource.onerror = () => {
      console.log('SSE connection error, will retry...');
    };

    return () => {
      eventSource.close();
    };
  }, [video?.status, videoId]);

  const fetchPlayers = async () => {
    try {
      const response = await fetch(`${API_URL}/api/players/video/${videoId}`);
      if (response.ok) {
        const data = await response.json();
        setPlayers(data);
      }
    } catch (err) {
      console.error('Failed to fetch players:', err);
    }
  };

  const fetchAllActions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/videos/${videoId}/actions`);
      if (response.ok) {
        const data = await response.json();
        setAllActions(data);
      }
    } catch (err) {
      console.error('Failed to fetch actions:', err);
    }
  };

  const fetchClips = async () => {
    try {
      const response = await fetch(`${API_URL}/api/clips/video/${videoId}`);
      if (response.ok) {
        const data = await response.json();
        setClips(data);
      }
    } catch (err) {
      console.error('Failed to fetch clips:', err);
    }
  };

  const deleteClip = async (clipId: string) => {
    setDeletingClipId(clipId);
    try {
      const response = await fetch(`${API_URL}/api/clips/${clipId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setClips(clips.filter(c => c.id !== clipId));
      }
    } catch (err) {
      console.error('Failed to delete clip:', err);
    } finally {
      setDeletingClipId(null);
    }
  };

  const fetchVideo = async () => {
    try {
      const response = await fetch(`${API_URL}/api/videos/${videoId}`);
      if (!response.ok) throw new Error('Video not found');
      const data = await response.json();
      setVideo(data);

      if (data.status === 'completed') {
        fetchPlayers();
        fetchAllActions();
        fetchClips();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load video');
    } finally {
      setLoading(false);
    }
  };

  // Fetch video data
  useEffect(() => {
    const fetchVideoData = async () => {
      try {
        const response = await fetch(`${API_URL}/api/videos/${videoId}`);
        if (!response.ok) throw new Error('Video not found');
        const data = await response.json();
        setVideo(data);

        // If still processing, poll for updates (fallback if SSE fails)
        if (data.status === 'processing' || data.status === 'queued') {
          const interval = setInterval(async () => {
            const res = await fetch(`${API_URL}/api/videos/${videoId}`);
            const updated = await res.json();
            setVideo(updated);
            if (updated.status === 'completed' || updated.status === 'failed') {
              clearInterval(interval);
              if (updated.status === 'completed') {
                fetchPlayers();
                fetchAllActions();
                fetchClips();
              }
            }
          }, 5000);
          return () => clearInterval(interval);
        } else if (data.status === 'completed') {
          fetchPlayers();
          fetchAllActions();
          fetchClips();
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load video');
      } finally {
        setLoading(false);
      }
    };

    fetchVideoData();
  }, [videoId]);

  // Fetch player timeline when selected
  useEffect(() => {
    if (!selectedPlayer) {
      setSegments([]);
      setActions([]);
      setMarkers([]);
      return;
    }

    const fetchTimeline = async () => {
      try {
        const response = await fetch(
          `${API_URL}/api/players/${selectedPlayer.id}/timeline`
        );
        if (response.ok) {
          const data = await response.json();
          setSegments(data.segments);
          setActions(data.actions);
          setMarkers(data.markers);
        }
      } catch (err) {
        console.error('Failed to fetch timeline:', err);
      }
    };

    fetchTimeline();
  }, [selectedPlayer]);

  // Video time update - depend on video status to re-run when video loads
  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement) return;

    const handleTimeUpdate = () => {
      setCurrentTime(videoElement.currentTime);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleLoadedMetadata = () => {
      console.log('Video metadata loaded, duration:', videoElement.duration);
    };
    const handleCanPlay = () => {
      console.log('Video can play');
    };
    const handleError = (e: Event) => {
      console.error('Video error:', e);
    };

    videoElement.addEventListener('timeupdate', handleTimeUpdate);
    videoElement.addEventListener('play', handlePlay);
    videoElement.addEventListener('pause', handlePause);
    videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
    videoElement.addEventListener('canplay', handleCanPlay);
    videoElement.addEventListener('error', handleError);

    return () => {
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
      videoElement.removeEventListener('play', handlePlay);
      videoElement.removeEventListener('pause', handlePause);
      videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
      videoElement.removeEventListener('canplay', handleCanPlay);
      videoElement.removeEventListener('error', handleError);
    };
  }, [video?.status]);

  const togglePlay = () => {
    const videoElement = videoRef.current;
    if (!videoElement) {
      console.error('Video element not found');
      return;
    }
    
    console.log('Toggle play, current state:', isPlaying, 'paused:', videoElement.paused);
    
    if (videoElement.paused) {
      videoElement.play().catch(err => {
        console.error('Error playing video:', err);
      });
    } else {
      videoElement.pause();
    }
  };

  const seekTo = (time: number) => {
    const videoElement = videoRef.current;
    if (!videoElement) {
      console.error('Video element not found');
      return;
    }
    console.log('Seeking to:', time);
    videoElement.currentTime = time;
  };

  const skipForward = () => {
    const videoElement = videoRef.current;
    if (!videoElement) return;
    videoElement.currentTime = Math.min(videoElement.currentTime + 10, videoElement.duration);
  };

  const skipBackward = () => {
    const videoElement = videoRef.current;
    if (!videoElement) return;
    videoElement.currentTime = Math.max(videoElement.currentTime - 10, 0);
  };

  const jumpToNextSegment = () => {
    const nextSegment = segments.find((s) => s.start_time > currentTime);
    if (nextSegment) {
      seekTo(nextSegment.start_time);
    }
  };

  const generateHighlights = async () => {
    if (!selectedPlayer) return;

    setGeneratingClips(true);
    try {
      const response = await fetch(
        `${API_URL}/api/clips/player/${selectedPlayer.id}/highlights`,
        { method: 'POST' }
      );
      if (response.ok) {
        const newClips = await response.json();
        // Refresh clips list to show the new clips
        await fetchClips();
        // Show success message with count
        if (newClips.length > 0) {
          // Scroll to clips section would be nice, but for now just update state
          console.log(`Generated ${newClips.length} highlight clips`);
        }
      } else {
        const error = await response.json();
        console.error('Failed to generate clips:', error);
      }
    } catch (err) {
      console.error('Failed to generate clips:', err);
    } finally {
      setGeneratingClips(false);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get icon and color for action type
  const getActionIcon = (actionType: string) => {
    switch (actionType) {
      case 'shot_attempt':
      case 'shot_made':
      case 'shot_missed':
      case 'three_point_attempt':
        return { icon: Circle, color: 'text-orange-500', bg: 'bg-orange-500/20' };
      case 'rebound':
        return { icon: Hand, color: 'text-blue-500', bg: 'bg-blue-500/20' };
      case 'assist':
        return { icon: ArrowRight, color: 'text-green-500', bg: 'bg-green-500/20' };
      case 'steal':
        return { icon: Zap, color: 'text-yellow-500', bg: 'bg-yellow-500/20' };
      case 'block':
        return { icon: XCircle, color: 'text-red-500', bg: 'bg-red-500/20' };
      case 'pass':
        return { icon: ArrowRight, color: 'text-purple-500', bg: 'bg-purple-500/20' };
      case 'turnover':
        return { icon: AlertCircle, color: 'text-red-400', bg: 'bg-red-400/20' };
      default:
        return { icon: Activity, color: 'text-gray-500', bg: 'bg-gray-500/20' };
    }
  };

  // Get unique action types for filter
  const actionTypes = [...new Set(allActions.map(a => a.action_type))];

  // Filter actions based on selected filter
  const filteredActions = actionFilter
    ? allActions.filter(a => a.action_type === actionFilter)
    : allActions;

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  if (error || !video) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-400">{error || 'Video not found'}</p>
        </div>
      </div>
    );
  }

  // Processing state
  if (video.status === 'processing' || video.status === 'queued') {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <Loader2 className="w-16 h-16 animate-spin text-primary-600 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
              Analyzing Video
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              {video.filename}
            </p>
          </div>

          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
              <span>{processingMessage || 'Processing...'}</span>
              <span>{processingProgress.toFixed(1)}%</span>
            </div>
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-300 ease-out"
                style={{ width: `${processingProgress}%` }}
              />
            </div>
            <div className="mt-2 text-xs text-gray-500 text-center">
              Stage: <span className="font-medium capitalize">{processingStage || video.status}</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-3 gap-4 mb-8">
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-center">
              <Eye className="w-6 h-6 text-blue-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {totalDetections.toLocaleString()}
              </div>
              <div className="text-xs text-gray-500">Detections</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-center">
              <User className="w-6 h-6 text-green-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {detectedPlayersCount}
              </div>
              <div className="text-xs text-gray-500">Players Found</div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 text-center">
              <Zap className="w-6 h-6 text-orange-500 mx-auto mb-2" />
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {detectedActionsCount}
              </div>
              <div className="text-xs text-gray-500">Actions Detected</div>
            </div>
          </div>

          {/* Live Detection Feed */}
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
            <div className="bg-gray-50 dark:bg-gray-700/50 px-4 py-2 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center">
                <Target className="w-4 h-4 mr-2 text-primary-500" />
                Live Detection Feed
              </h3>
            </div>
            <div className="h-48 overflow-y-auto bg-gray-900 p-3 font-mono text-xs">
              {liveDetections.length === 0 ? (
                <div className="text-gray-500 text-center py-8">
                  Waiting for detections...
                </div>
              ) : (
                liveDetections.map((item) => (
                  <div 
                    key={item.id}
                    className={`py-1 flex items-center space-x-2 ${
                      item.type === 'player' ? 'text-green-400' :
                      item.type === 'action' ? 'text-orange-400' :
                      'text-blue-400'
                    }`}
                  >
                    <span className="text-gray-500">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </span>
                    <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase ${
                      item.type === 'player' ? 'bg-green-500/20' :
                      item.type === 'action' ? 'bg-orange-500/20' :
                      'bg-blue-500/20'
                    }`}>
                      {item.type}
                    </span>
                    <span>{item.message}</span>
                    {item.confidence && (
                      <span className="text-gray-500">
                        ({(item.confidence * 100).toFixed(0)}%)
                      </span>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          <p className="text-xs text-gray-500 mt-4 text-center">
            This may take several minutes depending on video length...
          </p>

          {/* Cancel Button */}
          <div className="mt-6 flex justify-center">
            {showCancelConfirm ? (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-center">
                <p className="text-sm text-red-700 dark:text-red-400 mb-3">
                  Are you sure you want to cancel processing?
                </p>
                <div className="flex justify-center space-x-3">
                  <button
                    onClick={() => setShowCancelConfirm(false)}
                    className="px-4 py-2 text-sm bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    No, Continue
                  </button>
                  <button
                    onClick={cancelProcessing}
                    disabled={isCancelling}
                    className="px-4 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-red-400 flex items-center"
                  >
                    {isCancelling ? (
                      <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    ) : (
                      <XCircle className="w-4 h-4 mr-2" />
                    )}
                    Yes, Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setShowCancelConfirm(true)}
                className="px-4 py-2 text-sm bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 flex items-center"
              >
                <XCircle className="w-4 h-4 mr-2" />
                Cancel Processing
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Failed state
  if (video.status === 'failed') {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-red-200 dark:border-red-800 p-8 text-center">
          <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Processing Failed
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            {video.error_message || 'An error occurred while processing the video'}
          </p>
          <div className="flex justify-center space-x-3">
            <button
              onClick={retryProcessing}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry Processing
            </button>
            <a
              href="/"
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              Upload New Video
            </a>
          </div>
        </div>
      </div>
    );
  }

  // Cancelled state
  if (video.status === 'cancelled') {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-yellow-200 dark:border-yellow-800 p-8 text-center">
          <XCircle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
            Processing Cancelled
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Video processing was cancelled before completion.
          </p>
          <div className="flex justify-center space-x-3">
            <button
              onClick={retryProcessing}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Retry Processing
            </button>
            <a
              href="/"
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
            >
              Upload New Video
            </a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {video.filename}
          </h1>
          <div className="flex items-center space-x-4 mt-1 text-sm text-gray-600 dark:text-gray-400">
            <span className="flex items-center">
              <Clock className="w-4 h-4 mr-1" />
              {formatTime(video.duration_seconds)}
            </span>
            <span className="flex items-center">
              <User className="w-4 h-4 mr-1" />
              {video.player_count} players
            </span>
            <span className="flex items-center">
              <Activity className="w-4 h-4 mr-1" />
              {video.action_count} actions
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm flex items-center">
            <CheckCircle className="w-4 h-4 mr-1" />
            Analyzed
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video Player */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-black rounded-xl overflow-hidden">
            <video
              ref={videoRef}
              src={`${API_URL}/api/videos/${videoId}/stream`}
              className="w-full aspect-video"
              onClick={togglePlay}
              playsInline
              preload="auto"
            />
          </div>

          {/* Timeline */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            {/* Progress bar */}
            <div
              className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-lg cursor-pointer mb-4"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const percent = x / rect.width;
                seekTo(percent * video.duration_seconds);
              }}
            >
              {/* Segments */}
              {segments.map((segment) => (
                <div
                  key={segment.id}
                  className="absolute top-0 h-full bg-primary-500/40 rounded"
                  style={{
                    left: `${(segment.start_time / video.duration_seconds) * 100}%`,
                    width: `${((segment.end_time - segment.start_time) / video.duration_seconds) * 100}%`,
                  }}
                />
              ))}

              {/* Action markers - show all actions from play-by-play */}
              {allActions.map((action) => {
                const { color } = getActionIcon(action.action_type);
                const bgColor = color.replace('text-', 'bg-');
                return (
                  <div
                    key={action.id}
                    className={`absolute top-0 h-full w-1 ${bgColor} cursor-pointer hover:w-2 transition-all opacity-80 hover:opacity-100`}
                    style={{
                      left: `${(action.timestamp / video.duration_seconds) * 100}%`,
                    }}
                    title={`${action.action_type.replace('_', ' ')} by #${action.player_jersey || '?'} at ${formatTime(action.timestamp)}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      seekTo(action.timestamp);
                    }}
                  />
                );
              })}

              {/* Playhead */}
              <div
                className="absolute top-0 h-full w-1 bg-white shadow-lg"
                style={{
                  left: `${(currentTime / video.duration_seconds) * 100}%`,
                }}
              />
            </div>

            {/* Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <button
                  onClick={skipBackward}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                  title="Skip backward 10 seconds"
                >
                  <SkipBack className="w-5 h-5" />
                </button>
                <button
                  onClick={togglePlay}
                  className="p-3 bg-primary-600 hover:bg-primary-700 text-white rounded-full"
                  title={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? (
                    <Pause className="w-5 h-5" />
                  ) : (
                    <Play className="w-5 h-5" />
                  )}
                </button>
                <button
                  onClick={skipForward}
                  className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg"
                  title="Skip forward 10 seconds"
                >
                  <SkipForward className="w-5 h-5" />
                </button>
              </div>

              <div className="text-sm text-gray-600 dark:text-gray-400">
                {formatTime(currentTime)} / {formatTime(video.duration_seconds)}
              </div>

              {selectedPlayer && (
                <button
                  onClick={jumpToNextSegment}
                  className="px-3 py-1 text-sm bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-400 rounded-lg hover:bg-primary-200"
                >
                  Next #{selectedPlayer.jersey_number} moment
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Video Stats Summary */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
              <BarChart3 className="w-4 h-4 mr-2" />
              Analysis Summary
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-primary-600">{players.length}</div>
                <div className="text-xs text-gray-500">Players</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-orange-500">{allActions.length}</div>
                <div className="text-xs text-gray-500">Actions</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-green-500">{clips.length}</div>
                <div className="text-xs text-gray-500">Clips</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 text-center">
                <div className="text-2xl font-bold text-blue-500">
                  {video?.duration_seconds ? formatTime(video.duration_seconds) : '0:00'}
                </div>
                <div className="text-xs text-gray-500">Duration</div>
              </div>
            </div>
            
            {/* Action breakdown if any actions exist */}
            {allActions.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                <p className="text-xs text-gray-500 mb-2">Action breakdown:</p>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(
                    allActions.reduce((acc, a) => {
                      acc[a.action_type] = (acc[a.action_type] || 0) + 1;
                      return acc;
                    }, {} as Record<string, number>)
                  ).map(([type, count]) => {
                    const { color, bg } = getActionIcon(type);
                    return (
                      <span
                        key={type}
                        className={`px-2 py-1 text-xs rounded-full ${bg} ${color}`}
                      >
                        {type.replace('_', ' ')}: {count}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Player Selection */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              Detected Players
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {players.length === 0 ? (
                <p className="text-sm text-gray-500">No players detected</p>
              ) : (
                players.map((player) => (
                  <button
                    key={player.id}
                    onClick={() =>
                      setSelectedPlayer(
                        selectedPlayer?.id === player.id ? null : player
                      )
                    }
                    className={`w-full p-3 rounded-lg text-left transition-colors ${
                      selectedPlayer?.id === player.id
                        ? 'bg-primary-100 dark:bg-primary-900/30 border-primary-500'
                        : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                    } border`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {/* Team color indicator */}
                        {player.team_color && (
                          <div
                            className="w-3 h-3 rounded-full border border-gray-300"
                            style={{ backgroundColor: player.team_color }}
                            title={player.team || 'Team'}
                          />
                        )}
                        <span className="font-bold text-lg">
                          #{player.jersey_number || '?'}
                        </span>
                      </div>
                      <span className="text-xs text-gray-500">
                        {player.segment_count} segments
                      </span>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-xs text-gray-500">
                        {player.team || 'Unknown Team'}
                      </span>
                      <span className="text-xs text-gray-500">
                        {player.action_count} actions
                      </span>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Player Stats */}
          {selectedPlayer && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-2 mb-3">
                {selectedPlayer.team_color && (
                  <div
                    className="w-4 h-4 rounded-full border border-gray-300"
                    style={{ backgroundColor: selectedPlayer.team_color }}
                  />
                )}
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Player #{selectedPlayer.jersey_number} Stats
                </h3>
              </div>
              {selectedPlayer.team && (
                <p className="text-sm text-gray-500 mb-3">{selectedPlayer.team}</p>
              )}
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">
                    Time on screen
                  </span>
                  <span className="font-medium">
                    {formatTime(
                      segments.reduce(
                        (acc, s) => acc + (s.end_time - s.start_time),
                        0
                      )
                    )}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">
                    Segments
                  </span>
                  <span className="font-medium">{segments.length}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600 dark:text-gray-400">
                    Actions
                  </span>
                  <span className="font-medium">{actions.length}</span>
                </div>

                {/* Action breakdown */}
                {actions.length > 0 && (
                  <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
                    <p className="text-xs text-gray-500 mb-2">Action breakdown:</p>
                    {Object.entries(
                      actions.reduce((acc, a) => {
                        acc[a.action_type] = (acc[a.action_type] || 0) + 1;
                        return acc;
                      }, {} as Record<string, number>)
                    ).map(([type, count]) => (
                      <div
                        key={type}
                        className="flex justify-between text-xs py-1"
                      >
                        <span className="capitalize">{type.replace('_', ' ')}</span>
                        <span className="font-medium">{count}</span>
                      </div>
                    ))}
                  </div>
                )}

                <button
                  onClick={generateHighlights}
                  disabled={generatingClips}
                  className="w-full mt-4 py-2 px-4 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 text-white text-sm font-medium rounded-lg flex items-center justify-center"
                >
                  {generatingClips ? (
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  ) : (
                    <Download className="w-4 h-4 mr-2" />
                  )}
                  Generate Highlights
                </button>
              </div>
            </div>
          )}

          {/* Actions List (for selected player) */}
          {selectedPlayer && actions.length > 0 && (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                #{selectedPlayer.jersey_number} Actions
              </h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {actions.map((action) => {
                  const { icon: Icon, color, bg } = getActionIcon(action.action_type);
                  return (
                    <button
                      key={action.id}
                      onClick={() => seekTo(action.timestamp)}
                      className="w-full p-2 rounded-lg text-left bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <div className={`p-1 rounded ${bg}`}>
                            <Icon className={`w-3 h-3 ${color}`} />
                          </div>
                          <span className="text-sm font-medium capitalize">
                            {action.action_type.replace('_', ' ')}
                          </span>
                        </div>
                        <span className="text-xs text-gray-500">
                          {formatTime(action.timestamp)}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Play-by-Play - All Actions */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-900 dark:text-white flex items-center">
                <List className="w-4 h-4 mr-2" />
                Play-by-Play
              </h3>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-gray-500">{filteredActions.length} events</span>
                {actionFilter && (
                  <button
                    onClick={() => setActionFilter(null)}
                    className="text-xs text-primary-600 hover:text-primary-700"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            {/* Action Type Filters */}
            {actionTypes.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-3">
                {actionTypes.map((type) => {
                  const { color, bg } = getActionIcon(type);
                  const isActive = actionFilter === type;
                  return (
                    <button
                      key={type}
                      onClick={() => setActionFilter(isActive ? null : type)}
                      className={`px-2 py-1 text-xs rounded-full transition-colors ${
                        isActive
                          ? 'bg-primary-600 text-white'
                          : `${bg} ${color} hover:opacity-80`
                      }`}
                    >
                      {type.replace('_', ' ')}
                    </button>
                  );
                })}
              </div>
            )}

            {/* Actions List */}
            <div className="space-y-1 max-h-80 overflow-y-auto">
              {filteredActions.length === 0 ? (
                <div className="text-center py-4">
                  <Info className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-500 font-medium">
                    No basketball actions detected
                  </p>
                  <p className="text-xs text-gray-400 mt-1 px-2">
                    This may be due to the ball not being visible or tracked in the video.
                    Player segments are still available for highlight generation.
                  </p>
                </div>
              ) : (
                filteredActions.map((action) => {
                  const { icon: Icon, color, bg } = getActionIcon(action.action_type);
                  const isCurrentTime = Math.abs(currentTime - action.timestamp) < 0.5;
                  
                  return (
                    <button
                      key={action.id}
                      onClick={() => seekTo(action.timestamp)}
                      className={`w-full p-2 rounded-lg text-left transition-colors ${
                        isCurrentTime
                          ? 'bg-primary-100 dark:bg-primary-900/30 border border-primary-500'
                          : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        {/* Time */}
                        <span className="text-xs text-gray-500 w-10 flex-shrink-0">
                          {formatTime(action.timestamp)}
                        </span>
                        
                        {/* Icon */}
                        <div className={`p-1.5 rounded ${bg} flex-shrink-0`}>
                          <Icon className={`w-3 h-3 ${color}`} />
                        </div>
                        
                        {/* Player */}
                        <span className="text-sm font-bold text-gray-700 dark:text-gray-300 w-8 flex-shrink-0">
                          #{action.player_jersey || '?'}
                        </span>
                        
                        {/* Action Type */}
                        <span className="text-sm capitalize flex-grow truncate">
                          {action.action_type.replace('_', ' ')}
                        </span>
                        
                        {/* Confidence */}
                        <span className="text-xs text-gray-400 flex-shrink-0">
                          {action.confidence ? `${Math.round(action.confidence * 100)}%` : ''}
                        </span>
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>

          {/* Generated Clips Gallery */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-gray-900 dark:text-white flex items-center">
                <Film className="w-4 h-4 mr-2" />
                Generated Clips
              </h3>
              <span className="text-xs text-gray-500">{clips.length} clips</span>
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {clips.length === 0 ? (
                <div className="text-center py-4">
                  <Film className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">No clips generated yet</p>
                  <p className="text-xs text-gray-400 mt-1">
                    Select a player and click "Generate Highlights"
                  </p>
                </div>
              ) : (
                clips.map((clip) => (
                  <div
                    key={clip.id}
                    className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium truncate flex-grow">
                        {clip.title || 'Untitled Clip'}
                      </span>
                      <span className="text-xs text-gray-500 ml-2">
                        {clip.duration_seconds?.toFixed(1)}s
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-2">
                      <span>
                        {formatTime(clip.start_time)} - {formatTime(clip.end_time)}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => seekTo(clip.start_time)}
                        className="flex-1 px-2 py-1.5 bg-primary-600 hover:bg-primary-700 text-white text-xs rounded flex items-center justify-center"
                        title="Play in video"
                      >
                        <Play className="w-3 h-3 mr-1" />
                        Play
                      </button>
                      <a
                        href={`${API_URL}${clip.download_url}`}
                        download
                        className="flex-1 px-2 py-1.5 bg-green-600 hover:bg-green-700 text-white text-xs rounded flex items-center justify-center"
                        title="Download clip"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        Download
                      </a>
                      <button
                        onClick={() => deleteClip(clip.id)}
                        disabled={deletingClipId === clip.id}
                        className="px-2 py-1.5 bg-red-600 hover:bg-red-700 disabled:bg-red-400 text-white text-xs rounded flex items-center justify-center"
                        title="Delete clip"
                      >
                        {deletingClipId === clip.id ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Trash2 className="w-3 h-3" />
                        )}
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
