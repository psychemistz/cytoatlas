export function Footer() {
  return (
    <footer className="border-t border-border-light bg-bg-dark py-6 text-text-inverse">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4">
        <div className="text-sm text-text-muted">
          <p>&copy; 2025 CytoAtlas. Developed at NIH/NCI.</p>
          <p>Data from CIMA, Inflammation Atlas, and scAtlas.</p>
        </div>
        <div className="flex gap-4 text-sm">
          <a href="/docs" target="_blank" rel="noopener noreferrer" className="text-text-muted hover:text-text-inverse">
            API Documentation
          </a>
          <a
            href="https://github.com/psychemistz/cytoatlas"
            target="_blank"
            rel="noopener noreferrer"
            className="text-text-muted hover:text-text-inverse"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
