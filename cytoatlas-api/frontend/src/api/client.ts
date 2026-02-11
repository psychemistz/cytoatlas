import { API_BASE } from '@/lib/constants';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const token = localStorage.getItem('auth_token');
  if (token) {
    (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, { ...options, headers });

  if (!response.ok) {
    const body = await response.json().catch(() => ({ error: response.statusText }));
    throw new ApiError(response.status, body.detail || body.error || 'API request failed');
  }

  return response.json();
}

export function get<T>(endpoint: string, params?: Record<string, string>): Promise<T> {
  const qs = params ? `?${new URLSearchParams(params).toString()}` : '';
  return request<T>(`${endpoint}${qs}`, { method: 'GET' });
}

export function post<T>(endpoint: string, data: unknown): Promise<T> {
  return request<T>(endpoint, { method: 'POST', body: JSON.stringify(data) });
}
