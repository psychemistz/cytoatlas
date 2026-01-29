/**
 * CytoAtlas Client-Side Router
 * Simple hash-based or history API router for SPA navigation
 */

class Router {
    constructor() {
        this.routes = {};
        this.currentPage = null;
        this.params = {};

        // Bind methods
        this.handleRoute = this.handleRoute.bind(this);
    }

    /**
     * Register a route
     * @param {string} path - Route path (e.g., '/explore', '/atlas/:name')
     * @param {Function} handler - Handler function for the route
     */
    register(path, handler) {
        this.routes[path] = handler;
    }

    /**
     * Initialize the router
     */
    init() {
        // Listen for popstate (back/forward navigation)
        window.addEventListener('popstate', this.handleRoute);

        // Intercept link clicks for SPA navigation
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href]');
            if (!link) return;

            const href = link.getAttribute('href');

            // Skip external links and special links
            if (href.startsWith('http') || href.startsWith('//') ||
                href.startsWith('#') || href.startsWith('mailto:') ||
                link.target === '_blank') {
                return;
            }

            // Skip API and static routes
            if (href.startsWith('/api/') || href.startsWith('/static/') ||
                href.startsWith('/docs') || href.startsWith('/redoc')) {
                return;
            }

            e.preventDefault();
            this.navigate(href);
        });

        // Handle initial route
        this.handleRoute();
    }

    /**
     * Navigate to a path
     * @param {string} path - Path to navigate to
     * @param {boolean} replace - Replace current history entry
     */
    navigate(path, replace = false) {
        if (replace) {
            window.history.replaceState({}, '', path);
        } else {
            window.history.pushState({}, '', path);
        }
        this.handleRoute();
    }

    /**
     * Parse path and extract params
     * @param {string} routePath - Route pattern (e.g., '/atlas/:name')
     * @param {string} actualPath - Actual path (e.g., '/atlas/cima')
     * @returns {Object|null} Params object or null if no match
     */
    matchRoute(routePath, actualPath) {
        const routeParts = routePath.split('/').filter(Boolean);
        const pathParts = actualPath.split('/').filter(Boolean);

        if (routeParts.length !== pathParts.length) {
            return null;
        }

        const params = {};

        for (let i = 0; i < routeParts.length; i++) {
            const routePart = routeParts[i];
            const pathPart = pathParts[i];

            if (routePart.startsWith(':')) {
                // Parameter
                const paramName = routePart.slice(1);
                params[paramName] = decodeURIComponent(pathPart);
            } else if (routePart !== pathPart) {
                // No match
                return null;
            }
        }

        return params;
    }

    /**
     * Handle route change
     */
    handleRoute() {
        const path = window.location.pathname;
        const queryParams = new URLSearchParams(window.location.search);

        // Update active nav link
        this.updateNavLinks(path);

        // Try to match routes
        for (const [routePath, handler] of Object.entries(this.routes)) {
            const params = this.matchRoute(routePath, path);

            if (params !== null) {
                this.params = params;
                this.currentPage = routePath;

                // Call handler with params and query params
                handler(params, Object.fromEntries(queryParams));
                return;
            }
        }

        // No route matched - try home or 404
        if (path === '/' || path === '') {
            if (this.routes['/']) {
                this.routes['/']({}, Object.fromEntries(queryParams));
                return;
            }
        }

        // 404
        this.show404();
    }

    /**
     * Update navigation link active states
     */
    updateNavLinks(path) {
        document.querySelectorAll('.nav-link').forEach(link => {
            const href = link.getAttribute('href');
            const dataPage = link.getAttribute('data-page');

            link.classList.remove('active');

            // Match exact path or check if path starts with href (for nested routes)
            if (href === path || (href !== '/' && path.startsWith(href))) {
                link.classList.add('active');
            } else if (path === '/' && dataPage === 'home') {
                link.classList.add('active');
            }
        });
    }

    /**
     * Show 404 page
     */
    show404() {
        const app = document.getElementById('app');
        if (app) {
            app.innerHTML = `
                <div class="page" style="text-align: center; padding: 4rem;">
                    <h1>404 - Page Not Found</h1>
                    <p>The page you're looking for doesn't exist.</p>
                    <a href="/" class="btn btn-primary" style="margin-top: 1rem;">Go Home</a>
                </div>
            `;
        }
    }

    /**
     * Get current params
     */
    getParams() {
        return this.params;
    }

    /**
     * Get query params
     */
    getQueryParams() {
        return Object.fromEntries(new URLSearchParams(window.location.search));
    }
}

// Create global router instance
window.router = new Router();
