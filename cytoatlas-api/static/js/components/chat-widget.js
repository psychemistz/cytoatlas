/**
 * Chat Widget Component
 * Floating chat widget for quick access from any page
 */

const ChatWidget = {
    isOpen: false,
    isMinimized: true,
    conversationId: null,
    messages: [],
    isStreaming: false,
    sessionId: null,

    /**
     * Initialize the chat widget
     */
    init() {
        this.sessionId = this.getSessionId();
        this.createWidget();
        this.setupEventHandlers();
    },

    /**
     * Get or create session ID
     */
    getSessionId() {
        let sessionId = localStorage.getItem('cytoatlas_session');
        if (!sessionId) {
            sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('cytoatlas_session', sessionId);
        }
        return sessionId;
    },

    /**
     * Create the widget DOM
     */
    createWidget() {
        const widget = document.createElement('div');
        widget.id = 'chat-widget';
        widget.className = 'chat-widget minimized';
        widget.innerHTML = `
            <div class="widget-trigger" id="widget-trigger">
                <span class="trigger-icon">💬</span>
                <span class="trigger-text">Ask CytoAtlas</span>
            </div>

            <div class="widget-panel" id="widget-panel">
                <div class="widget-header">
                    <span class="widget-title">CytoAtlas Assistant</span>
                    <div class="widget-actions">
                        <button class="widget-btn expand-btn" id="expand-btn" title="Open full chat">
                            ↗
                        </button>
                        <button class="widget-btn minimize-btn" id="minimize-btn" title="Minimize">
                            −
                        </button>
                    </div>
                </div>

                <div class="widget-messages" id="widget-messages">
                    <div class="widget-welcome">
                        <p>Hi! Ask me anything about cytokine activity in CytoAtlas.</p>
                        <div class="quick-questions">
                            <button class="quick-q" data-q="What is IFNG activity in CD8 T cells?">
                                IFNG in CD8 T cells?
                            </button>
                            <button class="quick-q" data-q="Compare TNF across atlases">
                                Compare TNF
                            </button>
                            <button class="quick-q" data-q="Top cytokines in COVID-19?">
                                COVID-19 cytokines?
                            </button>
                        </div>
                    </div>
                </div>

                <div class="widget-input-area">
                    <input type="text"
                           id="widget-input"
                           placeholder="Ask a question..."
                           autocomplete="off">
                    <button id="widget-send" class="widget-send-btn" disabled>
                        ➤
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(widget);
    },

    /**
     * Set up event handlers
     */
    setupEventHandlers() {
        const trigger = document.getElementById('widget-trigger');
        const minimizeBtn = document.getElementById('minimize-btn');
        const expandBtn = document.getElementById('expand-btn');
        const input = document.getElementById('widget-input');
        const sendBtn = document.getElementById('widget-send');

        // Toggle widget
        trigger?.addEventListener('click', () => this.toggle());

        // Minimize
        minimizeBtn?.addEventListener('click', () => this.minimize());

        // Expand to full chat
        expandBtn?.addEventListener('click', () => {
            this.minimize();
            Router.navigate('/chat');
        });

        // Input handling
        input?.addEventListener('input', () => {
            sendBtn.disabled = !input.value.trim();
        });

        input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Send button
        sendBtn?.addEventListener('click', () => this.sendMessage());

        // Quick questions
        document.querySelectorAll('.quick-q').forEach(btn => {
            btn.addEventListener('click', () => {
                input.value = btn.dataset.q;
                sendBtn.disabled = false;
                this.sendMessage();
            });
        });
    },

    /**
     * Toggle widget open/closed
     */
    toggle() {
        this.isMinimized = !this.isMinimized;
        const widget = document.getElementById('chat-widget');
        widget?.classList.toggle('minimized', this.isMinimized);

        if (!this.isMinimized) {
            document.getElementById('widget-input')?.focus();
        }
    },

    /**
     * Minimize the widget
     */
    minimize() {
        this.isMinimized = true;
        document.getElementById('chat-widget')?.classList.add('minimized');
    },

    /**
     * Send a message
     */
    async sendMessage() {
        const input = document.getElementById('widget-input');
        const content = input?.value.trim();

        if (!content || this.isStreaming) return;

        // Clear input
        input.value = '';
        document.getElementById('widget-send').disabled = true;

        // Hide welcome
        const welcome = document.querySelector('.widget-welcome');
        welcome?.remove();

        const messagesContainer = document.getElementById('widget-messages');

        // Add user message
        const userDiv = document.createElement('div');
        userDiv.className = 'widget-message user';
        userDiv.textContent = content;
        messagesContainer.appendChild(userDiv);

        // Add assistant placeholder
        const assistantDiv = document.createElement('div');
        assistantDiv.className = 'widget-message assistant';
        assistantDiv.innerHTML = '<span class="typing">...</span>';
        messagesContainer.appendChild(assistantDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        this.isStreaming = true;

        try {
            const response = await fetch('/api/v1/chat/message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({
                    content: content,
                    conversation_id: this.conversationId,
                    session_id: this.sessionId,
                }),
            });

            if (!response.ok) {
                throw new Error('Request failed');
            }

            const data = await response.json();
            this.conversationId = data.conversation_id;

            // Update assistant message
            assistantDiv.textContent = data.content;

            // Add visualization indicators
            if (data.visualizations && data.visualizations.length > 0) {
                const vizNote = document.createElement('div');
                vizNote.className = 'viz-note';
                vizNote.innerHTML = `📊 ${data.visualizations.length} visualization(s) - <a href="/chat/${this.conversationId}">View in full chat</a>`;
                assistantDiv.appendChild(vizNote);
            }

            messagesContainer.scrollTop = messagesContainer.scrollHeight;

        } catch (error) {
            console.error('Widget chat error:', error);
            assistantDiv.textContent = 'Sorry, something went wrong. Please try again.';
        } finally {
            this.isStreaming = false;
        }
    },

    /**
     * Set context for the current page
     */
    setContext(context) {
        this.context = context;
    },
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ChatWidget.init());
} else {
    ChatWidget.init();
}

// Make available globally
window.ChatWidget = ChatWidget;
