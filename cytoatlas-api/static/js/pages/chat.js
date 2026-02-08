/**
 * Chat Page Handler
 * Full-page chat interface with CytoAtlas Assistant
 */

const ChatPage = {
    conversationId: null,
    sessionId: null,
    messages: [],
    isStreaming: false,
    suggestions: [],

    /**
     * Initialize the chat page
     */
    async init(params, query) {
        this.render();
        this.setupEventHandlers();

        // Get or create session ID
        this.sessionId = this.getSessionId();

        // Load suggestions
        await this.loadSuggestions();

        // Load conversation if ID provided
        if (params.conversationId) {
            await this.loadConversation(params.conversationId);
        }

        // Focus input
        document.getElementById('chat-input')?.focus();
    },

    /**
     * Render the chat page
     */
    render() {
        const app = document.getElementById('app');
        app.innerHTML = `
            <div class="chat-page">
                <aside class="chat-sidebar">
                    <div class="sidebar-header">
                        <h2>CytoAtlas Assistant</h2>
                        <button class="btn btn-primary btn-sm" id="new-chat-btn">
                            + New Chat
                        </button>
                    </div>
                    <div id="conversation-list" class="conversation-list">
                        <p class="loading">Loading conversations...</p>
                    </div>
                </aside>

                <main class="chat-main">
                    <div id="chat-messages" class="chat-messages">
                        <div class="chat-welcome" id="chat-welcome">
                            <div class="welcome-icon">🧬</div>
                            <h1>CytoAtlas Assistant</h1>
                            <p class="welcome-text">
                                Ask questions about cytokine and secreted protein activity
                                across 17+ million immune cells from three major atlases.
                            </p>

                            <div class="suggestions-section" id="suggestions-section">
                                <h3>Try asking:</h3>
                                <div class="suggestion-chips" id="suggestion-chips"></div>
                            </div>

                            <div class="disclaimer">
                                <strong>Note:</strong> Like other AI systems, CytoAtlas Assistant
                                can make mistakes. Key findings should be validated with
                                conventional bioinformatics approaches.
                            </div>
                        </div>
                    </div>

                    <div class="chat-input-container">
                        <div class="input-wrapper">
                            <textarea
                                id="chat-input"
                                placeholder="Ask about cytokine activity, cell types, diseases..."
                                rows="1"
                            ></textarea>
                            <button id="send-btn" class="send-btn" disabled>
                                <span class="send-icon">➤</span>
                            </button>
                        </div>
                        <div class="input-footer">
                            <span class="char-count" id="char-count">0 / 10000</span>
                            <span class="model-info">Powered by CytoAtlas LLM</span>
                        </div>
                    </div>
                </main>
            </div>
        `;

        this.loadConversationList();
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
     * Load suggestions
     */
    async loadSuggestions() {
        try {
            const response = await fetch('/api/v1/chat/suggestions');
            if (response.ok) {
                const data = await response.json();
                this.suggestions = data.suggestions;
                this.renderSuggestions();
            }
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
    },

    /**
     * Render suggestion chips
     */
    renderSuggestions() {
        const container = document.getElementById('suggestion-chips');
        if (!container) return;

        container.innerHTML = this.suggestions.map(s => `
            <button class="suggestion-chip ${s.category}" data-text="${s.text}">
                ${s.text}
            </button>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.suggestion-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.getElementById('chat-input').value = chip.dataset.text;
                this.updateSendButton();
                document.getElementById('chat-input').focus();
            });
        });
    },

    /**
     * Load conversation list
     */
    async loadConversationList() {
        const container = document.getElementById('conversation-list');

        try {
            const response = await fetch('/api/v1/chat/conversations', {
                credentials: 'include',
            });

            if (!response.ok) {
                container.innerHTML = '<p class="empty">Start a new chat!</p>';
                return;
            }

            const data = await response.json();

            if (data.conversations.length === 0) {
                container.innerHTML = '<p class="empty">No conversations yet</p>';
                return;
            }

            container.innerHTML = data.conversations.map(conv => `
                <div class="conversation-item ${conv.id === this.conversationId ? 'active' : ''}"
                     data-id="${conv.id}">
                    <div class="conv-title">${conv.title || 'Untitled'}</div>
                    <div class="conv-meta">${this.formatDate(conv.updated_at)}</div>
                </div>
            `).join('');

            // Add click handlers
            container.querySelectorAll('.conversation-item').forEach(item => {
                item.addEventListener('click', () => {
                    this.loadConversation(parseInt(item.dataset.id));
                });
            });

        } catch (error) {
            console.error('Failed to load conversations:', error);
            container.innerHTML = '<p class="error">Failed to load conversations</p>';
        }
    },

    /**
     * Load a specific conversation
     */
    async loadConversation(conversationId) {
        this.conversationId = conversationId;

        try {
            const response = await fetch(`/api/v1/chat/conversations/${conversationId}`, {
                credentials: 'include',
            });

            if (!response.ok) {
                throw new Error('Failed to load conversation');
            }

            const data = await response.json();
            this.messages = data.messages;
            this.renderMessages();

            // Update sidebar
            document.querySelectorAll('.conversation-item').forEach(item => {
                item.classList.toggle('active', parseInt(item.dataset.id) === conversationId);
            });

        } catch (error) {
            console.error('Failed to load conversation:', error);
        }
    },

    /**
     * Render all messages
     */
    renderMessages() {
        const container = document.getElementById('chat-messages');

        // Hide welcome if there are messages
        const welcome = document.getElementById('chat-welcome');
        if (this.messages.length > 0) {
            welcome?.remove();
        }

        // Clear existing messages (except welcome)
        container.querySelectorAll('.message').forEach(m => m.remove());

        // Render each message
        this.messages.forEach(msg => {
            container.appendChild(this.createMessageElement(msg));
        });

        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    },

    /**
     * Create a message element
     */
    createMessageElement(msg) {
        const div = document.createElement('div');
        div.className = `message ${msg.role}`;
        div.dataset.id = msg.id;

        let content = `<div class="message-content">${this.formatContent(msg.content)}</div>`;

        // Add visualizations
        if (msg.visualizations && msg.visualizations.length > 0) {
            content += '<div class="message-visualizations">';
            msg.visualizations.forEach(viz => {
                content += `<div class="viz-container" id="${viz.container_id}"></div>`;
            });
            content += '</div>';
        }

        // Add download button
        if (msg.downloadable_data) {
            content += `
                <div class="message-download">
                    <button class="btn btn-sm btn-outline download-btn"
                            data-message-id="${msg.id}">
                        📥 Download ${msg.downloadable_data.format.toUpperCase()}
                    </button>
                    <span class="download-desc">${msg.downloadable_data.description}</span>
                </div>
            `;
        }

        div.innerHTML = content;

        // Render visualizations after adding to DOM
        if (msg.visualizations) {
            setTimeout(() => {
                msg.visualizations.forEach(viz => {
                    this.renderVisualization(viz);
                });
            }, 0);
        }

        return div;
    },

    /**
     * Format message content (markdown-like)
     */
    formatContent(content) {
        if (!content) return '';

        // Basic markdown formatting
        return content
            // Code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Bold
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
            // Line breaks
            .replace(/\n/g, '<br>');
    },

    /**
     * Render a visualization
     */
    renderVisualization(viz) {
        const container = document.getElementById(viz.container_id);
        if (!container) return;

        // Use ChatViz component if available
        if (window.ChatViz) {
            ChatViz.render(container, viz);
        } else {
            // Fallback: show data as table
            container.innerHTML = `<pre>${JSON.stringify(viz.data, null, 2)}</pre>`;
        }
    },

    /**
     * Set up event handlers
     */
    setupEventHandlers() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const newChatBtn = document.getElementById('new-chat-btn');
        const charCount = document.getElementById('char-count');

        // Input handling
        input?.addEventListener('input', () => {
            this.updateSendButton();
            charCount.textContent = `${input.value.length} / 10000`;

            // Auto-resize
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
        });

        // Enter to send (Shift+Enter for newline)
        input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Send button
        sendBtn?.addEventListener('click', () => this.sendMessage());

        // New chat
        newChatBtn?.addEventListener('click', () => this.startNewChat());
    },

    /**
     * Update send button state
     */
    updateSendButton() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');

        const hasContent = input?.value.trim().length > 0;
        sendBtn.disabled = !hasContent || this.isStreaming;
    },

    /**
     * Send a message
     */
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const content = input?.value.trim();

        if (!content || this.isStreaming) return;

        // Clear input
        input.value = '';
        input.style.height = 'auto';
        this.updateSendButton();

        // Hide welcome
        document.getElementById('chat-welcome')?.remove();

        // Add user message
        const userMsg = {
            id: Date.now(),
            role: 'user',
            content: content,
            created_at: new Date().toISOString(),
        };
        this.messages.push(userMsg);

        const container = document.getElementById('chat-messages');
        container.appendChild(this.createMessageElement(userMsg));

        // Add streaming placeholder with thinking indicator
        const streamingDiv = document.createElement('div');
        streamingDiv.className = 'message assistant streaming';
        streamingDiv.innerHTML = `
            <div class="thinking-indicator">
                <div class="thinking-spinner"></div>
                <span class="thinking-text">Thinking...</span>
            </div>
            <div class="tool-status" style="display: none;"></div>
            <div class="message-content"></div>
        `;
        container.appendChild(streamingDiv);
        container.scrollTop = container.scrollHeight;

        this.isStreaming = true;
        this.updateSendButton();

        try {
            // Use streaming endpoint
            const response = await fetch('/api/v1/chat/message/stream', {
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
                throw new Error('Failed to send message');
            }

            // Read SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantContent = '';
            let visualizations = [];
            let messageId = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            // Use streamingDiv reference instead of IDs to avoid conflicts
                            const thinkingIndicator = streamingDiv.querySelector('.thinking-indicator');
                            const toolStatus = streamingDiv.querySelector('.tool-status');
                            const streamingContent = streamingDiv.querySelector('.message-content');

                            if (data.type === 'text') {
                                // Hide thinking indicator once we get text
                                if (thinkingIndicator) thinkingIndicator.style.display = 'none';
                                assistantContent += data.content;
                                if (streamingContent) {
                                    streamingContent.innerHTML = this.formatContent(assistantContent);
                                }
                            } else if (data.type === 'tool_call') {
                                // Show which tool is being called
                                const toolName = data.tool_call?.name || 'tool';
                                const toolLabels = {
                                    'search_entity': '🔍 Searching...',
                                    'get_activity_data': '📊 Fetching activity data...',
                                    'list_cell_types': '🧬 Getting cell types...',
                                    'list_signatures': '📋 Loading signatures...',
                                    'get_correlations': '📈 Analyzing correlations...',
                                    'get_disease_activity': '🏥 Getting disease data...',
                                    'compare_atlases': '⚖️ Comparing atlases...',
                                    'get_atlas_summary': '📑 Loading atlas summary...',
                                    'create_visualization': '📉 Creating visualization...',
                                    'export_data': '💾 Preparing export...',
                                };
                                const statusText = toolLabels[toolName] || `⚙️ Using ${toolName}...`;
                                if (toolStatus) {
                                    toolStatus.style.display = 'block';
                                    toolStatus.innerHTML = `<div class="tool-item">${statusText}</div>` + toolStatus.innerHTML;
                                }
                                if (thinkingIndicator) {
                                    thinkingIndicator.querySelector('.thinking-text').textContent = statusText;
                                }
                            } else if (data.type === 'tool_result') {
                                // Tool completed - could show checkmark
                                if (toolStatus && toolStatus.firstChild) {
                                    toolStatus.firstChild.innerHTML = toolStatus.firstChild.innerHTML.replace('...', ' ✓');
                                }
                            } else if (data.type === 'visualization') {
                                visualizations.push(data.visualization);
                            } else if (data.type === 'done') {
                                messageId = data.message_id;
                                if (thinkingIndicator) thinkingIndicator.style.display = 'none';
                                if (toolStatus) toolStatus.style.display = 'none';
                            } else if (data.type === 'error') {
                                throw new Error(data.content);
                            }

                            container.scrollTop = container.scrollHeight;
                        } catch (e) {
                            // Re-throw real errors, only ignore JSON parse errors
                            if (e.name !== 'SyntaxError') {
                                throw e;
                            }
                            // Ignore JSON parse errors for incomplete chunks
                        }
                    }
                }
            }

            // Finalize message
            streamingDiv.classList.remove('streaming');

            const assistantMsg = {
                id: messageId || Date.now() + 1,
                role: 'assistant',
                content: assistantContent,
                visualizations: visualizations,
                created_at: new Date().toISOString(),
            };
            this.messages.push(assistantMsg);

            // Update conversation ID from response
            if (!this.conversationId) {
                this.loadConversationList();
            }

            // Render visualizations
            visualizations.forEach(viz => {
                const vizContainer = document.createElement('div');
                vizContainer.className = 'viz-container';
                vizContainer.id = viz.container_id;
                streamingDiv.appendChild(vizContainer);
                this.renderVisualization(viz);
            });

        } catch (error) {
            console.error('Chat error:', error);
            streamingDiv.innerHTML = `
                <div class="message-content error">
                    Sorry, there was an error processing your request.
                    Please try again.
                </div>
            `;
        } finally {
            this.isStreaming = false;
            this.updateSendButton();
        }
    },

    /**
     * Start a new chat
     */
    startNewChat() {
        this.conversationId = null;
        this.messages = [];

        // Re-render
        this.render();
        this.loadSuggestions();
        this.setupEventHandlers();

        document.getElementById('chat-input')?.focus();
    },

    /**
     * Format date
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;

        return date.toLocaleDateString();
    },
};

// Make available globally
window.ChatPage = ChatPage;
