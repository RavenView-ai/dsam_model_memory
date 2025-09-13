/**
 * Tab Manager for JAM System
 * Handles chat state preservation without breaking normal navigation
 */

class TabManager {
    constructor() {
        this.chatState = null;
        this.inactivityTimer = null;
        this.inactivityTimeout = 30 * 60 * 1000; // 30 minutes
        this.lastActivity = Date.now();
        this.currentPath = window.location.pathname;
        this.init();
    }

    init() {
        // Only initialize if we're on the chat page
        if (this.isOnChatPage()) {
            this.saveChatState();
        }

        // Set up activity tracking
        this.setupActivityTracking();

        // Set up navigation listener for state preservation
        this.setupNavigationListener();

        // Restore state if returning to chat
        this.checkAndRestoreState();
    }

    isOnChatPage() {
        const path = window.location.pathname;
        return path === '/' || path === '/chat' || path === '/chat_basic';
    }

    setupActivityTracking() {
        // Track user activity
        ['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(event => {
            document.addEventListener(event, () => {
                this.lastActivity = Date.now();
                this.resetInactivityTimer();
            });
        });

        // Start inactivity timer only for chat page
        if (this.isOnChatPage()) {
            this.resetInactivityTimer();
        }
    }

    resetInactivityTimer() {
        if (this.inactivityTimer) {
            clearTimeout(this.inactivityTimer);
        }

        this.inactivityTimer = setTimeout(() => {
            this.handleInactivity();
        }, this.inactivityTimeout);
    }

    handleInactivity() {
        // Only clear chat state if we're on the chat page
        if (this.isOnChatPage()) {
            console.log('Clearing chat state due to inactivity');

            // Show notification
            const messagesContainer = document.querySelector('#chat-messages');
            if (messagesContainer) {
                const notification = document.createElement('div');
                notification.className = 'inactivity-notification';
                notification.innerHTML = `
                    <div style="
                        background: linear-gradient(135deg, rgba(255, 16, 240, 0.2), rgba(168, 85, 247, 0.2));
                        border: 1px solid var(--synth-pink);
                        border-radius: 8px;
                        padding: 1rem;
                        margin: 1rem 0;
                        color: var(--synth-text);
                        text-align: center;
                        animation: fadeIn 0.3s ease-in;
                    ">
                        <span style="color: var(--synth-cyan);">⚡</span>
                        Chat cleared due to inactivity (30 minutes)
                        <span style="color: var(--synth-cyan);">⚡</span>
                    </div>
                `;
                messagesContainer.appendChild(notification);

                // Clear the state after showing notification
                setTimeout(() => {
                    this.clearChatState();
                }, 3000);
            } else {
                this.clearChatState();
            }
        }
    }

    setupNavigationListener() {
        // Save state before leaving the chat page
        window.addEventListener('beforeunload', () => {
            if (this.isOnChatPage()) {
                this.saveChatState();
            }
        });

        // Listen for navigation clicks to save state
        document.addEventListener('click', (e) => {
            const link = e.target.closest('nav a');
            if (!link) return;

            // If we're on chat page and navigating away, save state
            if (this.isOnChatPage()) {
                this.saveChatState();
            }
        });

        // Handle browser back/forward
        window.addEventListener('popstate', () => {
            // Small delay to let the page load
            setTimeout(() => {
                if (this.isOnChatPage()) {
                    this.checkAndRestoreState();
                }
            }, 100);
        });
    }

    saveChatState() {
        const chatMessages = document.querySelector('#chat-messages');
        const inputField = document.querySelector('#chat-input') || document.querySelector('#message-input');
        const memoryToggle = document.querySelector('#memory-toggle');
        const sessionSelect = document.querySelector('#session-selector') || document.querySelector('#session-select');

        if (chatMessages) {
            // Don't save if chat is empty or only has system message
            const messages = chatMessages.querySelectorAll('.message');
            if (messages.length <= 1) {
                return;
            }

            this.chatState = {
                messages: chatMessages.innerHTML,
                input: inputField ? inputField.value : '',
                memoryEnabled: memoryToggle ? memoryToggle.checked : true,
                sessionId: sessionSelect ? sessionSelect.value : null,
                scrollPosition: chatMessages.scrollTop,
                timestamp: Date.now(),
                messageCount: messages.length
            };

            // Store in sessionStorage as backup
            try {
                sessionStorage.setItem('jam_chat_state', JSON.stringify(this.chatState));
            } catch (e) {
                console.warn('Failed to save chat state to sessionStorage:', e);
            }
        }
    }

    checkAndRestoreState() {
        // Only restore if we're on the chat page
        if (!this.isOnChatPage()) {
            return;
        }

        // Check if we have state to restore
        if (!this.chatState) {
            // Try to restore from sessionStorage
            const stored = sessionStorage.getItem('jam_chat_state');
            if (stored) {
                try {
                    this.chatState = JSON.parse(stored);
                } catch (e) {
                    console.error('Failed to parse stored chat state:', e);
                    return;
                }
            }
        }

        // Only restore if state is recent (within last hour)
        if (this.chatState && this.chatState.timestamp) {
            const age = Date.now() - this.chatState.timestamp;
            if (age > 60 * 60 * 1000) { // 1 hour
                console.log('Chat state is too old, not restoring');
                this.clearChatState();
                return;
            }
        }

        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this.restoreChatState();
            });
        } else {
            // Small delay to ensure chat elements are initialized
            setTimeout(() => {
                this.restoreChatState();
            }, 200);
        }
    }

    restoreChatState() {
        if (!this.chatState || !this.chatState.messages) {
            return;
        }

        const chatMessages = document.querySelector('#chat-messages');
        const inputField = document.querySelector('#chat-input') || document.querySelector('#message-input');
        const memoryToggle = document.querySelector('#memory-toggle');
        const sessionSelect = document.querySelector('#session-selector') || document.querySelector('#session-select');

        // Check if chat is already populated (avoid duplicate restoration)
        if (chatMessages) {
            const currentMessages = chatMessages.querySelectorAll('.message');
            if (currentMessages.length > 1 && currentMessages.length >= (this.chatState.messageCount || 0)) {
                console.log('Chat already has content, skipping restoration');
                return;
            }

            // Restore messages
            chatMessages.innerHTML = this.chatState.messages;
            chatMessages.scrollTop = this.chatState.scrollPosition || 0;

            // Show restoration indicator
            this.showRestorationIndicator();
        }

        if (inputField && this.chatState.input) {
            inputField.value = this.chatState.input;
        }

        if (memoryToggle && typeof this.chatState.memoryEnabled !== 'undefined') {
            memoryToggle.checked = this.chatState.memoryEnabled;
        }

        if (sessionSelect && this.chatState.sessionId) {
            sessionSelect.value = this.chatState.sessionId;
        }
    }

    showRestorationIndicator() {
        const indicator = document.getElementById('chat-state-indicator');
        if (indicator) {
            indicator.style.display = 'flex';
            indicator.style.opacity = '0.8';

            // Auto-hide after 5 seconds
            setTimeout(() => {
                if (indicator) {
                    indicator.style.opacity = '0.3';
                    setTimeout(() => {
                        indicator.style.display = 'none';
                    }, 300);
                }
            }, 5000);
        }
    }

    clearChatState() {
        this.chatState = null;
        sessionStorage.removeItem('jam_chat_state');

        // Only clear UI if we're on the chat page
        if (this.isOnChatPage()) {
            const chatMessages = document.querySelector('#chat-messages');
            if (chatMessages) {
                // Keep system message if exists
                const systemMessage = chatMessages.querySelector('.message.system');
                if (systemMessage) {
                    chatMessages.innerHTML = '';
                    chatMessages.appendChild(systemMessage);
                } else {
                    chatMessages.innerHTML = `
                        <div class="message system">
                            <div class="message-content">
                                Welcome to JAM Chat. Your conversation has been cleared.
                            </div>
                        </div>
                    `;
                }
            }

            const inputField = document.querySelector('#chat-input') || document.querySelector('#message-input');
            if (inputField) {
                inputField.value = '';
            }

            // Hide state indicator
            const indicator = document.getElementById('chat-state-indicator');
            if (indicator) {
                indicator.style.display = 'none';
            }
        }
    }

    // Public API for manual state management
    saveState() {
        if (this.isOnChatPage()) {
            this.saveChatState();
        }
    }

    clearState() {
        this.clearChatState();
    }

    setInactivityTimeout(minutes) {
        this.inactivityTimeout = minutes * 60 * 1000;
        if (this.isOnChatPage()) {
            this.resetInactivityTimer();
        }
    }
}

// Initialize tab manager when DOM is ready
let tabManager = null;

function initTabManager() {
    if (!tabManager) {
        tabManager = new TabManager();
        window.tabManager = tabManager;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTabManager);
} else {
    initTabManager();
}

// Re-initialize when navigating back to the page
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        initTabManager();
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TabManager;
}