/**
 * Methods Page Handler
 */

const MethodsPage = {
    /**
     * Initialize the methods page
     */
    async init() {
        this.render();
    },

    /**
     * Render the methods page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('methods-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }
    },
};

// Make available globally
window.MethodsPage = MethodsPage;
