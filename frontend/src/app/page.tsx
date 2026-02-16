'use client';

import { useState, useCallback } from 'react';
import { Upload, Link, Loader2, CheckCircle, AlertCircle, Settings, Users } from 'lucide-react';
import { useRouter } from 'next/navigation';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const router = useRouter();
  const [uploadMode, setUploadMode] = useState<'file' | 'youtube'>('file');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [dragActive, setDragActive] = useState(false);
  
  // Analysis options
  const [showOptions, setShowOptions] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<'full' | 'targeted'>('full');
  const [targetJersey, setTargetJersey] = useState('');
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [homeColor, setHomeColor] = useState('');
  const [awayColor, setAwayColor] = useState('');

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        setUploadMode('file');
      }
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    setUploading(true);
    setUploadStatus('idle');
    setErrorMessage('');

    try {
      const formData = new FormData();

      if (uploadMode === 'file' && selectedFile) {
        formData.append('file', selectedFile);
      } else if (uploadMode === 'youtube' && youtubeUrl) {
        formData.append('youtube_url', youtubeUrl);
      } else {
        throw new Error('Please select a file or enter a YouTube URL');
      }

      // Add analysis options
      formData.append('analysis_mode', analysisMode);
      if (analysisMode === 'targeted' && targetJersey) {
        formData.append('target_jersey', targetJersey);
      }
      if (homeTeam) formData.append('home_team', homeTeam);
      if (awayTeam) formData.append('away_team', awayTeam);
      if (homeColor) formData.append('home_color', homeColor);
      if (awayColor) formData.append('away_color', awayColor);

      const response = await fetch(`${API_URL}/api/videos/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
      }

      const data = await response.json();
      setUploadStatus('success');

      // Redirect to video analysis page after short delay
      setTimeout(() => {
        router.push(`/analyze/${data.id}`);
      }, 1500);
    } catch (error) {
      setUploadStatus('error');
      setErrorMessage(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Basketball Video Analysis
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Upload a video or paste a YouTube URL to analyze player movements and actions
        </p>
      </div>

      {/* Upload Mode Toggle */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex rounded-lg border border-gray-200 dark:border-gray-700 p-1 bg-white dark:bg-gray-800">
          <button
            onClick={() => setUploadMode('file')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              uploadMode === 'file'
                ? 'bg-primary-600 text-white'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Upload className="w-4 h-4 inline-block mr-2" />
            Upload File
          </button>
          <button
            onClick={() => setUploadMode('youtube')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              uploadMode === 'youtube'
                ? 'bg-primary-600 text-white'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            <Link className="w-4 h-4 inline-block mr-2" />
            YouTube URL
          </button>
        </div>
      </div>

      {/* Upload Area */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        {uploadMode === 'file' ? (
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                : 'border-gray-300 dark:border-gray-600 hover:border-primary-400'
            }`}
          >
            <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
            <p className="text-gray-600 dark:text-gray-400 mb-2">
              Drag and drop your video here, or
            </p>
            <label className="inline-block">
              <input
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <span className="cursor-pointer text-primary-600 hover:text-primary-700 font-medium">
                browse files
              </span>
            </label>
            <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
              Supports MP4, MOV, AVI, MKV, WEBM (max 2GB)
            </p>

            {selectedFile && (
              <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-gray-500">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                YouTube Video URL
              </label>
              <input
                type="url"
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=..."
                className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-500">
              Paste a YouTube video URL to download and analyze
            </p>
          </div>
        )}

        {/* Analysis Options Toggle */}
        <button
          onClick={() => setShowOptions(!showOptions)}
          className="mt-4 w-full py-2 px-4 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white flex items-center justify-center border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        >
          <Settings className="w-4 h-4 mr-2" />
          {showOptions ? 'Hide' : 'Show'} Analysis Options
        </button>

        {/* Analysis Options Panel */}
        {showOptions && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg space-y-4">
            {/* Analysis Mode */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Analysis Mode
              </label>
              <div className="flex space-x-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="analysisMode"
                    value="full"
                    checked={analysisMode === 'full'}
                    onChange={() => setAnalysisMode('full')}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Full Analysis (all players)
                  </span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    name="analysisMode"
                    value="targeted"
                    checked={analysisMode === 'targeted'}
                    onChange={() => setAnalysisMode('targeted')}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700 dark:text-gray-300">
                    Target Specific Player (faster)
                  </span>
                </label>
              </div>
            </div>

            {/* Target Player (if targeted mode) */}
            {analysisMode === 'targeted' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Target Jersey Number
                </label>
                <input
                  type="text"
                  value={targetJersey}
                  onChange={(e) => setTargetJersey(e.target.value)}
                  placeholder="e.g., 23"
                  className="w-32 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Only track this player for faster processing
                </p>
              </div>
            )}

            {/* Team Information */}
            <div className="pt-4 border-t border-gray-200 dark:border-gray-600">
              <div className="flex items-center mb-3">
                <Users className="w-4 h-4 mr-2 text-gray-500" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Team Information (Optional)
                </span>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Home Team Name</label>
                  <input
                    type="text"
                    value={homeTeam}
                    onChange={(e) => setHomeTeam(e.target.value)}
                    placeholder="e.g., Lakers"
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Home Jersey Color</label>
                  <input
                    type="text"
                    value={homeColor}
                    onChange={(e) => setHomeColor(e.target.value)}
                    placeholder="e.g., Yellow"
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Away Team Name</label>
                  <input
                    type="text"
                    value={awayTeam}
                    onChange={(e) => setAwayTeam(e.target.value)}
                    placeholder="e.g., Celtics"
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-500 mb-1">Away Jersey Color</label>
                  <input
                    type="text"
                    value={awayColor}
                    onChange={(e) => setAwayColor(e.target.value)}
                    placeholder="e.g., Green"
                    className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm"
                  />
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Team info helps identify which team each player belongs to
              </p>
            </div>
          </div>
        )}

        {/* Status Messages */}
        {uploadStatus === 'success' && (
          <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg flex items-center">
            <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
            <span className="text-green-700 dark:text-green-400">
              Upload successful! Redirecting to analysis...
            </span>
          </div>
        )}

        {uploadStatus === 'error' && (
          <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
            <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
            <span className="text-red-700 dark:text-red-400">{errorMessage}</span>
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={
            uploading ||
            (uploadMode === 'file' && !selectedFile) ||
            (uploadMode === 'youtube' && !youtubeUrl)
          }
          className="mt-6 w-full py-3 px-4 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center"
        >
          {uploading ? (
            <>
              <Loader2 className="w-5 h-5 mr-2 animate-spin" />
              {uploadMode === 'youtube' ? 'Downloading...' : 'Uploading...'}
            </>
          ) : (
            <>
              <Upload className="w-5 h-5 mr-2" />
              {analysisMode === 'targeted' && targetJersey 
                ? `Analyze Player #${targetJersey}` 
                : 'Analyze Video'}
            </>
          )}
        </button>
      </div>

      {/* Features */}
      <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="w-10 h-10 bg-primary-100 dark:bg-primary-900/30 rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Player Tracking
          </h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            Automatically detect and track players throughout the video with jersey number recognition
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Action Recognition
          </h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            Detect shots, rebounds, assists, and other basketball actions automatically
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="w-10 h-10 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Highlight Clips
          </h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm">
            Generate highlight clips for specific players or key moments in the game
          </p>
        </div>
      </div>
    </div>
  );
}
