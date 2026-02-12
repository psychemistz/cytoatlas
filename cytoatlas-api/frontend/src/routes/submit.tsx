import { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router';
import { get, post } from '@/api/client';
import { API_BASE } from '@/lib/constants';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Job {
  id: number;
  atlas_name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_step?: string;
  created_at: string;
  n_cells?: number;
  n_samples?: number;
  n_cell_types?: number;
  error_message?: string;
}

interface UploadInitResponse {
  upload_id: string;
  chunk_size: number;
  total_chunks: number;
}

interface UploadCompleteResponse {
  file_path: string;
}

interface ProcessResponse {
  job_id: number;
}

type UploadState = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024; // 50 GB

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function statusColor(status: Job['status']): string {
  switch (status) {
    case 'pending':
      return 'bg-yellow-100 text-yellow-800';
    case 'processing':
      return 'bg-blue-100 text-blue-800';
    case 'completed':
      return 'bg-green-100 text-green-800';
    case 'failed':
      return 'bg-red-100 text-red-800';
    case 'cancelled':
      return 'bg-gray-100 text-gray-600';
    default:
      return 'bg-gray-100 text-gray-600';
  }
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function Submit() {
  // Auth gate removed: server-side RBAC handles permissions; no login UI exists
  const isAuthenticated = true;

  const [file, setFile] = useState<File | null>(null);
  const [atlasName, setAtlasName] = useState('');
  const [description, setDescription] = useState('');
  const [signatureTypes, setSignatureTypes] = useState({ cytosig: true, secact: true });
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState('');
  const [jobId, setJobId] = useState<number | null>(null);
  const [jobStatus, setJobStatus] = useState<Job | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cancelledRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* ---- Fetch user jobs ---- */
  const fetchJobs = useCallback(async () => {
    try {
      const data = await get<Job[]>('/submit/jobs');
      setJobs(data);
    } catch {
      /* auth may be missing; ignore */
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) fetchJobs();
  }, [isAuthenticated, fetchJobs]);

  /* ---- WebSocket / polling for active job ---- */
  useEffect(() => {
    if (!jobId || uploadState === 'completed' || uploadState === 'failed' || uploadState === 'idle') return;

    let wsConnected = false;

    function connectWebSocket() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}${API_BASE}/ws/jobs/${jobId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        wsConnected = true;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as Partial<Job>;
          setJobStatus((prev) => ({ ...prev, ...data } as Job));
          if (data.progress !== undefined) setUploadProgress(data.progress);
          if (data.current_step) setUploadStatus(data.current_step);
          if (data.status === 'completed') {
            setUploadState('completed');
            fetchJobs();
          } else if (data.status === 'failed') {
            setUploadState('failed');
            setUploadStatus(data.error_message || 'Processing failed');
            fetchJobs();
          }
        } catch {
          /* ignore malformed messages */
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        wsConnected = false;
        // Fallback to polling if the job is still active
        if (uploadState === 'processing') startPolling();
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    function startPolling() {
      if (pollingRef.current) return;
      pollingRef.current = setInterval(async () => {
        try {
          const data = await get<Job>(`/submit/jobs/${jobId}`);
          setJobStatus(data);
          if (data.progress !== undefined) setUploadProgress(data.progress);
          if (data.current_step) setUploadStatus(data.current_step);
          if (data.status === 'completed') {
            setUploadState('completed');
            fetchJobs();
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = null;
          } else if (data.status === 'failed') {
            setUploadState('failed');
            setUploadStatus(data.error_message || 'Processing failed');
            fetchJobs();
            if (pollingRef.current) clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
        } catch {
          /* ignore transient errors */
        }
      }, 3000);
    }

    connectWebSocket();
    // If WS does not connect within 3 seconds, start polling
    const fallbackTimer = setTimeout(() => {
      if (!wsConnected) startPolling();
    }, 3000);

    return () => {
      clearTimeout(fallbackTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [jobId, uploadState, fetchJobs]);

  /* ---- File selection ---- */
  function handleFileSelect(selected: File | null) {
    if (!selected) return;
    if (!selected.name.endsWith('.h5ad')) {
      setUploadStatus('Only .h5ad files are accepted');
      return;
    }
    if (selected.size > MAX_FILE_SIZE) {
      setUploadStatus(`File exceeds maximum size of ${formatFileSize(MAX_FILE_SIZE)}`);
      return;
    }
    setFile(selected);
    setUploadStatus('');
    setUploadState('idle');
  }

  /* ---- Drag and drop ---- */
  function handleDragOver(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFileSelect(dropped);
  }

  /* ---- Chunked upload + process ---- */
  async function handleSubmit() {
    if (!file || !atlasName.trim()) return;
    cancelledRef.current = false;

    try {
      setUploadState('uploading');
      setUploadProgress(0);
      setUploadStatus('Initializing upload...');

      // Step 1: Init
      const init = await post<UploadInitResponse>('/submit/upload/init', {
        filename: file.name,
        file_size: file.size,
      });

      const { upload_id, chunk_size, total_chunks } = init;

      // Step 2: Upload chunks
      for (let i = 0; i < total_chunks; i++) {
        if (cancelledRef.current) {
          setUploadState('idle');
          setUploadStatus('Upload cancelled');
          return;
        }

        const start = i * chunk_size;
        const end = Math.min(start + chunk_size, file.size);
        const chunk = file.slice(start, end);

        const formData = new FormData();
        formData.append('upload_id', upload_id);
        formData.append('chunk_index', String(i));
        formData.append('chunk', chunk);

        const token = localStorage.getItem('auth_token');
        const headers: Record<string, string> = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const res = await fetch(`${API_BASE}/submit/upload/chunk`, {
          method: 'POST',
          headers,
          body: formData,
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({ detail: 'Chunk upload failed' }));
          throw new Error(body.detail || 'Chunk upload failed');
        }

        const pct = Math.round(((i + 1) / total_chunks) * 100);
        setUploadProgress(pct);
        setUploadStatus(`Uploading chunk ${i + 1}/${total_chunks}...`);
      }

      if (cancelledRef.current) {
        setUploadState('idle');
        setUploadStatus('Upload cancelled');
        return;
      }

      // Step 3: Complete upload
      setUploadStatus('Finalizing upload...');
      const complete = await post<UploadCompleteResponse>('/submit/upload/complete', {
        upload_id,
      });

      // Step 4: Start processing
      setUploadState('processing');
      setUploadProgress(0);
      setUploadStatus('Starting processing...');

      const sigTypes: string[] = [];
      if (signatureTypes.cytosig) sigTypes.push('cytosig');
      if (signatureTypes.secact) sigTypes.push('secact');

      const processResult = await post<ProcessResponse>('/submit/process', {
        file_path: complete.file_path,
        atlas_name: atlasName.trim(),
        description: description.trim() || undefined,
        signature_types: sigTypes,
      });

      setJobId(processResult.job_id);
      setJobStatus({
        id: processResult.job_id,
        atlas_name: atlasName.trim(),
        status: 'pending',
        progress: 0,
        created_at: new Date().toISOString(),
      });
    } catch (err) {
      setUploadState('failed');
      setUploadStatus(err instanceof Error ? err.message : 'Upload failed');
    }
  }

  function handleCancel() {
    cancelledRef.current = true;
    setUploadState('idle');
    setUploadStatus('Upload cancelled');
    setUploadProgress(0);
  }

  /* ---- Auth gate ---- */
  if (!isAuthenticated) {
    return (
      <div className="mx-auto max-w-[1400px] px-4 py-12">
        <div className="mx-auto max-w-md rounded-xl border border-border-light bg-bg-secondary p-8 text-center shadow-sm">
          <h2 className="mb-2 text-xl font-bold">Login Required</h2>
          <p className="mb-6 text-sm text-text-secondary">
            You must be logged in to submit datasets for processing.
          </p>
          <Link
            to="/login"
            className="inline-block rounded-lg bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary/90"
          >
            Go to Login
          </Link>
        </div>
      </div>
    );
  }

  /* ---- Main render ---- */
  return (
    <div className="mx-auto max-w-[1400px] px-4 py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Submit Your Dataset</h1>
        <p className="text-text-secondary">
          Upload your single-cell H5AD file for CytoSig/SecAct activity inference
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {/* Left column: upload + options */}
        <div className="lg:col-span-2">
          {/* File Upload Zone */}
          <div className="mb-6 rounded-xl border border-border-light p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Upload H5AD File</h2>
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
                dragOver
                  ? 'border-primary bg-primary/5'
                  : file
                    ? 'border-green-400 bg-green-50/50'
                    : 'border-border-light hover:border-primary/50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".h5ad"
                className="hidden"
                onChange={(e) => handleFileSelect(e.target.files?.[0] ?? null)}
              />
              {file ? (
                <div>
                  <p className="mb-1 text-sm font-medium">{file.name}</p>
                  <p className="text-xs text-text-muted">{formatFileSize(file.size)}</p>
                  <button
                    type="button"
                    className="mt-2 text-xs text-primary hover:underline"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                      setUploadState('idle');
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    }}
                  >
                    Change file
                  </button>
                </div>
              ) : (
                <div>
                  <p className="mb-1 text-sm text-text-secondary">
                    Drag and drop your .h5ad file here, or click to browse
                  </p>
                  <button
                    type="button"
                    className="mt-2 rounded-lg bg-primary px-4 py-1.5 text-sm font-medium text-white hover:bg-primary/90"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                  >
                    Select File
                  </button>
                  <p className="mt-2 text-xs text-text-muted">Max size: 50 GB</p>
                </div>
              )}
            </div>
          </div>

          {/* Options Section */}
          {file && uploadState === 'idle' && (
            <div className="mb-6 rounded-xl border border-border-light p-6 shadow-sm">
              <h2 className="mb-4 text-lg font-semibold">Processing Options</h2>
              <div className="space-y-4">
                <div>
                  <label htmlFor="atlas-name" className="mb-1 block text-sm font-medium">
                    Atlas Name <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="atlas-name"
                    type="text"
                    value={atlasName}
                    onChange={(e) => setAtlasName(e.target.value)}
                    placeholder="e.g., My PBMC Dataset"
                    className="w-full rounded-lg border border-border-light bg-bg-secondary px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
                <div>
                  <label htmlFor="description" className="mb-1 block text-sm font-medium">
                    Description
                  </label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    placeholder="Brief description of your dataset (optional)"
                    rows={3}
                    className="w-full rounded-lg border border-border-light bg-bg-secondary px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
                <div>
                  <span className="mb-2 block text-sm font-medium">Signature Types</span>
                  <div className="flex gap-6">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={signatureTypes.cytosig}
                        onChange={(e) =>
                          setSignatureTypes((s) => ({ ...s, cytosig: e.target.checked }))
                        }
                        className="rounded border-border-light text-primary focus:ring-primary"
                      />
                      CytoSig (44 cytokines)
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={signatureTypes.secact}
                        onChange={(e) =>
                          setSignatureTypes((s) => ({ ...s, secact: e.target.checked }))
                        }
                        className="rounded border-border-light text-primary focus:ring-primary"
                      />
                      SecAct (1,249 proteins)
                    </label>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={!atlasName.trim() || (!signatureTypes.cytosig && !signatureTypes.secact)}
                  className="rounded-lg bg-primary px-6 py-2 text-sm font-medium text-white hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Upload &amp; Process
                </button>
              </div>
            </div>
          )}

          {/* Upload Progress */}
          {uploadState === 'uploading' && (
            <div className="mb-6 rounded-xl border border-border-light p-6 shadow-sm">
              <h2 className="mb-4 text-lg font-semibold">Uploading</h2>
              <div className="mb-2 h-3 w-full overflow-hidden rounded-full bg-gray-200">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <div className="mb-4 flex items-center justify-between text-sm">
                <span className="text-text-secondary">{uploadStatus}</span>
                <span className="font-medium">{uploadProgress}%</span>
              </div>
              <button
                type="button"
                onClick={handleCancel}
                className="rounded-lg border border-red-300 px-4 py-1.5 text-sm font-medium text-red-600 hover:bg-red-50"
              >
                Cancel
              </button>
            </div>
          )}

          {/* Processing Status */}
          {(uploadState === 'processing' || uploadState === 'completed' || uploadState === 'failed') &&
            jobStatus && (
              <div className="mb-6 rounded-xl border border-border-light p-6 shadow-sm">
                <h2 className="mb-4 text-lg font-semibold">Processing Status</h2>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <span
                      className={`inline-block rounded-full px-3 py-0.5 text-xs font-medium ${statusColor(jobStatus.status)}`}
                    >
                      {jobStatus.status.charAt(0).toUpperCase() + jobStatus.status.slice(1)}
                    </span>
                    {jobId !== null && (
                      <span className="text-xs text-text-muted">Job ID: {jobId}</span>
                    )}
                  </div>

                  {uploadState === 'processing' && (
                    <>
                      <div className="h-3 w-full overflow-hidden rounded-full bg-gray-200">
                        <div
                          className="h-full rounded-full bg-blue-500 transition-all duration-300"
                          style={{ width: `${uploadProgress}%` }}
                        />
                      </div>
                      <p className="text-sm text-text-secondary">{uploadStatus}</p>
                    </>
                  )}

                  {uploadState === 'failed' && (
                    <p className="text-sm text-red-600">{uploadStatus}</p>
                  )}

                  {jobStatus.status === 'completed' && (
                    <>
                      <div className="flex flex-wrap gap-4 text-sm">
                        {jobStatus.n_cells != null && (
                          <span>
                            <strong>{jobStatus.n_cells.toLocaleString()}</strong> cells
                          </span>
                        )}
                        {jobStatus.n_samples != null && (
                          <span>
                            <strong>{jobStatus.n_samples.toLocaleString()}</strong> samples
                          </span>
                        )}
                        {jobStatus.n_cell_types != null && (
                          <span>
                            <strong>{jobStatus.n_cell_types.toLocaleString()}</strong> cell types
                          </span>
                        )}
                      </div>
                      <Link
                        to={`/atlas/${jobStatus.atlas_name}`}
                        className="inline-block rounded-lg bg-primary px-4 py-1.5 text-sm font-medium text-white hover:bg-primary/90"
                      >
                        View Atlas
                      </Link>
                    </>
                  )}
                </div>
              </div>
            )}
        </div>

        {/* Right column: jobs list */}
        <div>
          <div className="rounded-xl border border-border-light p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Your Jobs</h2>
            {jobs.length === 0 ? (
              <p className="text-sm text-text-muted">No jobs yet. Upload a dataset to get started.</p>
            ) : (
              <div className="space-y-3">
                {jobs.map((job) => (
                  <div
                    key={job.id}
                    className="rounded-lg border border-border-light p-3"
                  >
                    <div className="mb-1 flex items-center justify-between">
                      <span className="text-sm font-medium">{job.atlas_name}</span>
                      <span
                        className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${statusColor(job.status)}`}
                      >
                        {job.status}
                      </span>
                    </div>
                    <p className="mb-2 text-xs text-text-muted">{formatDate(job.created_at)}</p>
                    {(job.status === 'pending' || job.status === 'processing') && (
                      <div className="mb-2">
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-200">
                          <div
                            className="h-full rounded-full bg-blue-500 transition-all duration-300"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                        {job.current_step && (
                          <p className="mt-1 text-xs text-text-muted">{job.current_step}</p>
                        )}
                      </div>
                    )}
                    {job.status === 'completed' && (
                      <Link
                        to={`/atlas/${job.atlas_name}`}
                        className="text-xs font-medium text-primary hover:underline"
                      >
                        View Atlas
                      </Link>
                    )}
                    {job.status === 'failed' && job.error_message && (
                      <p className="text-xs text-red-500">{job.error_message}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
