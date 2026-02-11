import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { get, post, ApiError } from '@/api/client';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  mockFetch.mockReset();
  localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ApiError', () => {
  it('stores status and message', () => {
    const error = new ApiError(404, 'Not found');
    expect(error.status).toBe(404);
    expect(error.message).toBe('Not found');
    expect(error.name).toBe('ApiError');
    expect(error).toBeInstanceOf(Error);
  });
});

describe('get', () => {
  it('fetches from the correct URL', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ data: 'test' }),
    });

    const result = await get('/atlases');
    expect(result).toEqual({ data: 'test' });
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/v1/atlases',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('appends query params', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve([]),
    });

    await get('/atlases/cima/activity', { signature_type: 'CytoSig' });
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toBe('/api/v1/atlases/cima/activity?signature_type=CytoSig');
  });

  it('includes auth token when present', async () => {
    localStorage.setItem('auth_token', 'test-jwt-token');
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    });

    await get('/protected');
    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['Authorization']).toBe('Bearer test-jwt-token');
  });

  it('throws ApiError on non-OK response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Server Error',
      json: () => Promise.resolve({ detail: 'Internal error' }),
    });

    try {
      await get('/fail');
      expect.fail('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(ApiError);
      expect((e as ApiError).status).toBe(500);
      expect((e as ApiError).message).toBe('Internal error');
    }
  });

  it('handles non-JSON error response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      json: () => Promise.reject(new Error('not json')),
    });

    await expect(get('/fail')).rejects.toThrow('Bad Gateway');
  });
});

describe('post', () => {
  it('sends JSON body', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ id: 1 }),
    });

    const result = await post('/submit', { name: 'test' });
    expect(result).toEqual({ id: 1 });

    const [, options] = mockFetch.mock.calls[0];
    expect(options.method).toBe('POST');
    expect(options.body).toBe(JSON.stringify({ name: 'test' }));
    expect(options.headers['Content-Type']).toBe('application/json');
  });
});
