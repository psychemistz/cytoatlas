/**
 * About Page Handler
 */

const AboutPage = {
    /**
     * Initialize the about page
     */
    async init() {
        this.render();
    },

    /**
     * Render the about page template
     */
    render() {
        const app = document.getElementById('app');
        const template = document.getElementById('about-template');

        if (app && template) {
            app.innerHTML = template.innerHTML;
        }
    },
};

// Make available globally
window.AboutPage = AboutPage;
