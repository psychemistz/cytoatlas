import { Link } from 'react-router';

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-4 text-center">
      <h1 className="mb-2 text-6xl font-bold text-text-muted">404</h1>
      <p className="mb-6 text-lg text-text-secondary">Page not found</p>
      <Link to="/" className="rounded-lg bg-primary px-6 py-2 font-medium text-text-inverse hover:bg-primary-dark">
        Back to Home
      </Link>
    </div>
  );
}
