export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface ApiStatus {
  status: string;
  version: string;
  uptime: number;
}
