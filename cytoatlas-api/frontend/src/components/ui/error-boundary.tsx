import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="rounded-lg border border-danger/20 bg-danger/5 p-6 text-center">
            <p className="mb-2 font-medium text-danger">Something went wrong</p>
            <p className="text-sm text-text-secondary">{this.state.error?.message}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="mt-4 rounded-md bg-primary px-4 py-2 text-sm text-text-inverse hover:bg-primary-dark"
            >
              Try again
            </button>
          </div>
        )
      );
    }
    return this.props.children;
  }
}
