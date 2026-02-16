'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Video,
  Clock,
  User,
  Activity,
  Loader2,
  CheckCircle,
  AlertCircle,
  Trash2,
  XCircle,
  RefreshCw,
  StopCircle,
  Filter,
  ArrowUpDown,
} from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface VideoItem {
  id: string;
  filename: string;
  status: string;
  duration_seconds: number;
  player_count: number;
  action_count: number;
  created_at: string;
  error_message?: string;
}

type SortOption = 'date_desc' | 'date_asc' | 'name_asc' | 'name_desc' | 'duration_desc' | 'duration_asc';
type StatusFilter = 'all' | 'completed' | 'processing' | 'failed' | 'cancelled';

export default function VideosPage() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [sortOption, setSortOption] = useState<SortOption>('date_desc');

  useEffect(() => {
    fetchVideos();
    
    // Poll for status updates every 5 seconds
    const interval = setInterval(() => {
      fetchVideos();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchVideos = async () => {
    try {
      const response = await fetch(`${API_URL}/api/videos/`);
      if (response.ok) {
        const data = await response.json();
        setVideos(data);
      }
    } catch (err) {
      console.error('Failed to fetch videos:', err);
    } finally {
      setLoading(false);
    }
  };

  const deleteVideo = async (id: string) => {
    if (!confirm('Are you sure you want to delete this video? This cannot be undone.')) return;

    setActionLoading(id);
    try {
      const response = await fetch(`${API_URL}/api/videos/${id}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setVideos(videos.filter((v) => v.id !== id));
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to delete video');
      }
    } catch (err) {
      console.error('Failed to delete video:', err);
      alert('Failed to delete video');
    } finally {
      setActionLoading(null);
    }
  };

  const cancelVideo = async (id: string) => {
    if (!confirm('Are you sure you want to cancel processing this video?')) return;

    setActionLoading(id);
    try {
      const response = await fetch(`${API_URL}/api/videos/${id}/cancel`, {
        method: 'POST',
      });
      if (response.ok) {
        fetchVideos();
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to cancel');
      }
    } catch (err) {
      console.error('Failed to cancel video:', err);
      alert('Failed to cancel video');
    } finally {
      setActionLoading(null);
    }
  };

  const retryVideo = async (id: string) => {
    setActionLoading(id);
    try {
      const formData = new FormData();
      const response = await fetch(`${API_URL}/api/videos/${id}/retry`, {
        method: 'POST',
        body: formData,
      });
      if (response.ok) {
        fetchVideos();
      } else {
        const data = await response.json();
        alert(data.detail || 'Failed to retry');
      }
    } catch (err) {
      console.error('Failed to retry video:', err);
      alert('Failed to retry video');
    } finally {
      setActionLoading(null);
    }
  };

  // Filter and sort videos
  const filteredAndSortedVideos = videos
    .filter((video) => {
      if (statusFilter === 'all') return true;
      if (statusFilter === 'processing') return video.status === 'processing' || video.status === 'queued';
      return video.status === statusFilter;
    })
    .sort((a, b) => {
      switch (sortOption) {
        case 'date_desc':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        case 'date_asc':
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        case 'name_asc':
          return a.filename.localeCompare(b.filename);
        case 'name_desc':
          return b.filename.localeCompare(a.filename);
        case 'duration_desc':
          return (b.duration_seconds || 0) - (a.duration_seconds || 0);
        case 'duration_asc':
          return (a.duration_seconds || 0) - (b.duration_seconds || 0);
        default:
          return 0;
      }
    });

  // Count videos by status
  const statusCounts = {
    all: videos.length,
    completed: videos.filter(v => v.status === 'completed').length,
    processing: videos.filter(v => v.status === 'processing' || v.status === 'queued').length,
    failed: videos.filter(v => v.status === 'failed').length,
    cancelled: videos.filter(v => v.status === 'cancelled').length,
  };

  const formatTime = (seconds: number) => {
    if (!seconds) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-xs flex items-center">
            <CheckCircle className="w-3 h-3 mr-1" />
            Completed
          </span>
        );
      case 'processing':
      case 'queued':
        return (
          <span className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-xs flex items-center">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            {status === 'processing' ? 'Processing' : 'Queued'}
          </span>
        );
      case 'cancelling':
        return (
          <span className="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded-full text-xs flex items-center">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Cancelling
          </span>
        );
      case 'cancelled':
        return (
          <span className="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded-full text-xs flex items-center">
            <XCircle className="w-3 h-3 mr-1" />
            Cancelled
          </span>
        );
      case 'failed':
        return (
          <span className="px-2 py-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded-full text-xs flex items-center">
            <AlertCircle className="w-3 h-3 mr-1" />
            Failed
          </span>
        );
      default:
        return (
          <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full text-xs">
            {status}
          </span>
        );
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          My Videos
        </h1>
        <Link
          href="/"
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium"
        >
          Upload New Video
        </Link>
      </div>

      {/* Filters and Sort */}
      {videos.length > 0 && (
        <div className="flex flex-wrap items-center gap-4 mb-6">
          {/* Status Filter */}
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
              aria-label="Filter by status"
              className="px-3 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All ({statusCounts.all})</option>
              <option value="completed">Completed ({statusCounts.completed})</option>
              <option value="processing">Processing ({statusCounts.processing})</option>
              <option value="failed">Failed ({statusCounts.failed})</option>
              <option value="cancelled">Cancelled ({statusCounts.cancelled})</option>
            </select>
          </div>

          {/* Sort */}
          <div className="flex items-center space-x-2">
            <ArrowUpDown className="w-4 h-4 text-gray-500" />
            <select
              value={sortOption}
              onChange={(e) => setSortOption(e.target.value as SortOption)}
              aria-label="Sort videos"
              className="px-3 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg text-sm focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="date_desc">Newest First</option>
              <option value="date_asc">Oldest First</option>
              <option value="name_asc">Name (A-Z)</option>
              <option value="name_desc">Name (Z-A)</option>
              <option value="duration_desc">Longest First</option>
              <option value="duration_asc">Shortest First</option>
            </select>
          </div>

          {/* Results count */}
          <span className="text-sm text-gray-500">
            Showing {filteredAndSortedVideos.length} of {videos.length} videos
          </span>
        </div>
      )}

      {videos.length === 0 ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl p-12 text-center border border-gray-200 dark:border-gray-700">
          <Video className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
            No videos yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Upload your first basketball video to get started
          </p>
          <Link
            href="/"
            className="inline-block px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm font-medium"
          >
            Upload Video
          </Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {filteredAndSortedVideos.length === 0 ? (
            <div className="bg-white dark:bg-gray-800 rounded-xl p-8 text-center border border-gray-200 dark:border-gray-700">
              <p className="text-gray-600 dark:text-gray-400">
                No videos match the current filter.
              </p>
              <button
                onClick={() => setStatusFilter('all')}
                className="mt-2 text-primary-600 hover:text-primary-700 text-sm"
              >
                Clear filter
              </button>
            </div>
          ) : (
            filteredAndSortedVideos.map((video) => (
            <div
              key={video.id}
              className="bg-white dark:bg-gray-800 rounded-xl p-4 shadow-sm border border-gray-200 dark:border-gray-700 hover:border-primary-300 dark:hover:border-primary-700 transition-colors"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <Link
                    href={`/analyze/${video.id}`}
                    className="block group"
                  >
                    <h3 className="font-medium text-gray-900 dark:text-white truncate group-hover:text-primary-600">
                      {video.filename}
                    </h3>
                    <div className="flex items-center space-x-4 mt-1 text-sm text-gray-500">
                      <span className="flex items-center">
                        <Clock className="w-4 h-4 mr-1" />
                        {formatTime(video.duration_seconds)}
                      </span>
                      {video.status === 'completed' && (
                        <>
                          <span className="flex items-center">
                            <User className="w-4 h-4 mr-1" />
                            {video.player_count} players
                          </span>
                          <span className="flex items-center">
                            <Activity className="w-4 h-4 mr-1" />
                            {video.action_count} actions
                          </span>
                        </>
                      )}
                      <span>{formatDate(video.created_at)}</span>
                    </div>
                  </Link>
                </div>

                <div className="flex items-center space-x-2 ml-4">
                  {getStatusBadge(video.status)}
                  
                  {/* Cancel button for processing/queued videos */}
                  {(video.status === 'processing' || video.status === 'queued') && (
                    <button
                      onClick={() => cancelVideo(video.id)}
                      disabled={actionLoading === video.id}
                      className="p-2 text-gray-400 hover:text-yellow-600 hover:bg-yellow-50 dark:hover:bg-yellow-900/20 rounded-lg transition-colors disabled:opacity-50"
                      title="Cancel processing"
                    >
                      {actionLoading === video.id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <StopCircle className="w-4 h-4" />
                      )}
                    </button>
                  )}
                  
                  {/* Retry button for failed/cancelled videos */}
                  {(video.status === 'failed' || video.status === 'cancelled') && (
                    <button
                      onClick={() => retryVideo(video.id)}
                      disabled={actionLoading === video.id}
                      className="p-2 text-gray-400 hover:text-primary-600 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-lg transition-colors disabled:opacity-50"
                      title="Retry processing"
                    >
                      {actionLoading === video.id ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <RefreshCw className="w-4 h-4" />
                      )}
                    </button>
                  )}
                  
                  {/* Delete button */}
                  <button
                    onClick={() => deleteVideo(video.id)}
                    disabled={actionLoading === video.id}
                    className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors disabled:opacity-50"
                    title="Delete video"
                  >
                    {actionLoading === video.id ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>

              {video.status === 'failed' && video.error_message && (
                <div className="mt-3 p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <p className="text-xs text-red-600 dark:text-red-400">
                    {video.error_message}
                  </p>
                </div>
              )}
            </div>
          ))
          )}
        </div>
      )}
    </div>
  );
}
