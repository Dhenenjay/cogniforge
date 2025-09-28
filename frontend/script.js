/**
 * CogniForge Frontend Script
 * Handles SSE connections, real-time logging, and chart updates
 */

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
    API_BASE: 'http://localhost:8001',  // Updated port for working API
    SSE_RECONNECT_DELAY: 3000,
    CHART_MAX_POINTS: 50,
    LOG_MAX_LINES: 1000,
    UPDATE_INTERVAL: 100,  // ms between chart updates
    ANIMATION_SPEED: 0.01,
    COLORS: {
        primary: '#2563eb',
        secondary: '#10b981',
        danger: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6',
        success: '#10b981',
        error: '#ef4444',
        dim: '#9ca3af'
    }
};

// ============================================================================
// Global State
// ============================================================================

class AppState {
    constructor() {
        this.eventSource = null;
        this.currentRequestId = null;
        this.executionStartTime = null;
        this.timerInterval = null;
        this.animationFrame = null;
        this.isExecuting = false;
        
        // Data buffers
        this.lossData = [];
        this.rewardData = [];
        this.eventQueue = [];
        this.logBuffer = [];
        
        // Chart instances
        this.charts = {
            loss: null,
            reward: null
        };
        
        // Metrics
        this.metrics = {
            bcLoss: null,
            reward: null,
            stage: null,
            progress: 0
        };
    }
    
    reset() {
        this.lossData = [];
        this.rewardData = [];
        this.eventQueue = [];
        this.logBuffer = [];
        this.metrics = {
            bcLoss: null,
            reward: null,
            stage: null,
            progress: 0
        };
    }
}

const appState = new AppState();

// ============================================================================
// SSE Event Management
// ============================================================================

class SSEManager {
    constructor() {
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }
    
    /**
     * Connect to SSE endpoint for real-time events
     */
    connect(requestId) {
        this.disconnect();  // Close any existing connection
        
        const url = `${CONFIG.API_BASE}/events/${requestId}`;
        console.log(`[SSE] Connecting to: ${url}`);
        
        try {
            appState.eventSource = new EventSource(url);
            
            // Connection opened
            appState.eventSource.onopen = (event) => {
                console.log('[SSE] Connection established');
                this.reconnectAttempts = 0;
                Logger.log('Connected to event stream', 'success');
            };
            
            // Message received
            appState.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleEvent(data);
                } catch (error) {
                    console.error('[SSE] Failed to parse event:', error);
                    Logger.log(`Invalid event data: ${error.message}`, 'error');
                }
            };
            
            // Error occurred
            appState.eventSource.onerror = (error) => {
                console.error('[SSE] Connection error:', error);
                
                if (appState.eventSource.readyState === EventSource.CLOSED) {
                    Logger.log('Event stream connection closed', 'warning');
                    this.handleDisconnection();
                } else if (appState.eventSource.readyState === EventSource.CONNECTING) {
                    Logger.log('Reconnecting to event stream...', 'info');
                }
            };
            
        } catch (error) {
            console.error('[SSE] Failed to create EventSource:', error);
            Logger.log(`Failed to connect: ${error.message}`, 'error');
        }
    }
    
    /**
     * Disconnect from SSE
     */
    disconnect() {
        if (appState.eventSource) {
            appState.eventSource.close();
            appState.eventSource = null;
            console.log('[SSE] Disconnected');
        }
    }
    
    /**
     * Handle SSE event
     */
    handleEvent(data) {
        // Add to event queue for processing
        appState.eventQueue.push(data);
        
        // Process immediately if not batching
        this.processEvent(data);
    }
    
    /**
     * Process individual event
     */
    processEvent(data) {
        // Update progress if available
        if (data.progress !== undefined && data.progress >= 0) {
            UI.updateProgress(data.progress * 100, data.message);
        }
        
        // Handle different event types
        if (data.type === 'heartbeat') {
            this.handleHeartbeat(data);
        } else if (data.phase) {
            this.handlePhaseEvent(data);
        }
        
        // Check for completion
        if (data.phase === 'completed' || data.phase === 'failed') {
            this.handleCompletion(data.phase === 'completed');
        }
    }
    
    /**
     * Handle heartbeat event
     */
    handleHeartbeat(data) {
        // Log heartbeat with dimmed style
        if (data.phase && data.message) {
            Logger.log(`[${data.phase}] ${data.message}`, 'heartbeat');
        }
        
        // Update metrics from heartbeat
        if (data.metrics) {
            MetricsManager.updateFromHeartbeat(data.phase, data.metrics);
        }
    }
    
    /**
     * Handle phase change event
     */
    handlePhaseEvent(data) {
        const phaseIcons = {
            'connected': '🔌',
            'planning': '📋',
            'expert_demonstration': '👨‍🏫',
            'behavior_cloning': '🧠',
            'optimization': '⚙️',
            'vision_refinement': '👁️',
            'code_generation': '💻',
            'execution': '🤖',
            'completed': '✅',
            'failed': '❌',
            'stream_end': '🏁'
        };
        
        const icon = phaseIcons[data.phase] || '●';
        
        // Log phase change
        Logger.log(`${icon} [${data.phase}] ${data.message}`, 'phase');
        
        // Update current stage display
        UI.updateStage(data.phase);
        
        // Update phase banner
        if (data.phase !== 'connected' && data.phase !== 'stream_end') {
            phaseBanner.updatePhase(data.phase, 'active');
        }
        
        // Log metrics if available
        if (data.metrics) {
            this.logMetrics(data.metrics);
            MetricsManager.updateFromEvent(data.phase, data.metrics);
        }
    }
    
    /**
     * Log metrics to console
     */
    logMetrics(metrics) {
        for (const [key, value] of Object.entries(metrics)) {
            if (key === 'request_id') continue;  // Skip request_id
            
            let formattedValue = value;
            if (typeof value === 'number') {
                formattedValue = value.toFixed(4);
            } else if (typeof value === 'boolean') {
                formattedValue = value ? '✓' : '✗';
            }
            
            Logger.log(`  ${key}: ${formattedValue}`, 'metric');
        }
    }
    
    /**
     * Handle completion
     */
    handleCompletion(success) {
        this.disconnect();
        
        if (success) {
            UI.setStatus('success');
            Logger.log('✅ Task completed successfully!', 'success');
            UI.updateProgress(100, 'Execution complete');
            phaseBanner.updatePhase('completed', 'completed');
            Animation.showSuccess();
        } else {
            UI.setStatus('error');
            Logger.log('❌ Task execution failed', 'error');
            phaseBanner.updatePhase('failed', 'failed');
        }
        
        UI.resetControls();
        Timer.stop();
    }
    
    /**
     * Handle disconnection
     */
    handleDisconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts && appState.isExecuting) {
            this.reconnectAttempts++;
            Logger.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}...`, 'warning');
            
            setTimeout(() => {
                if (appState.currentRequestId) {
                    this.connect(appState.currentRequestId);
                }
            }, CONFIG.SSE_RECONNECT_DELAY);
        } else {
            Logger.log('Connection lost. Please check the server.', 'error');
            UI.resetControls();
        }
    }
}

// ============================================================================
// Logging System
// ============================================================================

class Logger {
    static log(message, type = 'info') {
        // Add to buffer
        appState.logBuffer.push({
            message,
            type,
            timestamp: new Date()
        });
        
        // Trim buffer if too large
        if (appState.logBuffer.length > CONFIG.LOG_MAX_LINES) {
            appState.logBuffer.shift();
        }
        
        // Append to console
        this.appendToConsole(message, type);
    }
    
    static appendToConsole(message, type) {
        const consoleEl = document.getElementById('console');
        if (!consoleEl) return;
        
        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        line.textContent = `[${timestamp}] ${message}`;
        
        // Style based on type
        switch(type) {
            case 'phase':
                line.style.color = '#bc8cff';
                line.style.fontWeight = 'bold';
                line.style.marginTop = '10px';
                break;
            case 'metric':
                line.style.color = '#79c0ff';
                line.style.paddingLeft = '20px';
                break;
            case 'heartbeat':
                line.style.color = '#6b7280';
                line.style.fontSize = '11px';
                break;
            case 'success':
                line.style.color = '#56d364';
                break;
            case 'error':
                line.style.color = '#f85149';
                break;
            case 'warning':
                line.style.color = '#d29922';
                break;
            default:
                line.style.color = '#58a6ff';
        }
        
        consoleEl.appendChild(line);
        
        // Auto-scroll to bottom
        consoleEl.scrollTop = consoleEl.scrollHeight;
        
        // Limit console lines
        while (consoleEl.children.length > CONFIG.LOG_MAX_LINES) {
            consoleEl.removeChild(consoleEl.firstChild);
        }
    }
    
    static clear() {
        const consoleEl = document.getElementById('console');
        if (consoleEl) {
            consoleEl.innerHTML = '';
        }
        appState.logBuffer = [];
    }
}

// ============================================================================
// Metrics Management
// ============================================================================

class MetricsManager {
    /**
     * Update metrics from heartbeat event
     */
    static updateFromHeartbeat(phase, metrics) {
        // BC Training metrics
        if (phase === 'behavior_cloning') {
            if (metrics.epoch_loss !== undefined) {
                appState.lossData.push({
                    epoch: metrics.epoch || appState.lossData.length + 1,
                    loss: metrics.epoch_loss
                });
                ChartManager.updateLossChart();
                UI.updateMetric('bcLoss', metrics.epoch_loss);
            }
            
            if (metrics.current_loss !== undefined) {
                // Could add batch-level tracking here
            }
        }
        
        // Optimization metrics
        if (phase === 'optimization') {
            if (metrics.avg_reward !== undefined) {
                appState.rewardData.push({
                    step: metrics.step || appState.rewardData.length + 1,
                    reward: metrics.avg_reward
                });
                ChartManager.updateRewardChart();
                UI.updateMetric('reward', metrics.avg_reward);
            }
            
            if (metrics.current_reward !== undefined) {
                // Track individual rewards if needed
            }
        }
        
        // Vision metrics
        if (phase === 'vision_refinement') {
            if (metrics.aligned !== undefined) {
                Logger.log(`Vision alignment: ${metrics.aligned ? 'Yes' : 'No'}`, 'info');
            }
            if (metrics.offset_mm !== undefined) {
                Logger.log(`Vision offset: ${metrics.offset_mm.toFixed(2)}mm`, 'metric');
            }
        }
    }
    
    /**
     * Update metrics from phase event
     */
    static updateFromEvent(phase, metrics) {
        if (metrics.bc_loss !== undefined) {
            UI.updateMetric('bcLoss', metrics.bc_loss);
        }
        
        if (metrics.final_reward !== undefined) {
            UI.updateMetric('reward', metrics.final_reward);
        }
        
        if (metrics.planning_time !== undefined) {
            Logger.log(`Planning completed in ${metrics.planning_time.toFixed(2)}s`, 'metric');
        }
        
        if (metrics.num_trajectories !== undefined) {
            Logger.log(`Collected ${metrics.num_trajectories} expert demonstrations`, 'metric');
        }
    }
    
    /**
     * Clear all metrics
     */
    static clear() {
        appState.lossData = [];
        appState.rewardData = [];
        UI.updateMetric('bcLoss', null);
        UI.updateMetric('reward', null);
        ChartManager.clearCharts();
    }
}

// ============================================================================
// Chart Management
// ============================================================================

class ChartManager {
    /**
     * Initialize charts
     */
    static initialize() {
        this.initializeLossChart();
        this.initializeRewardChart();
    }
    
    /**
     * Initialize BC Loss chart
     */
    static initializeLossChart() {
        const ctx = document.getElementById('lossChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (appState.charts.loss) {
            appState.charts.loss.destroy();
            appState.charts.loss = null;
        }
        
        appState.charts.loss = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'BC Loss',
                    data: [],
                    borderColor: CONFIG.COLORS.danger,
                    backgroundColor: `${CONFIG.COLORS.danger}20`,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: CONFIG.COLORS.danger,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: CONFIG.COLORS.dim,
                        borderColor: CONFIG.COLORS.primary,
                        borderWidth: 1,
                        displayColors: false,
                        callbacks: {
                            label: (context) => {
                                return `Loss: ${context.parsed.y.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 11
                            },
                            callback: (value) => value.toFixed(3)
                        },
                        grid: {
                            color: 'rgba(55, 65, 81, 0.3)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Loss',
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 12
                            }
                        }
                    },
                    x: {
                        ticks: {
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 11
                            },
                            maxRotation: 45,
                            minRotation: 0
                        },
                        grid: {
                            color: 'rgba(55, 65, 81, 0.3)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Epoch',
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }
    
    /**
     * Initialize Reward chart
     */
    static initializeRewardChart() {
        const ctx = document.getElementById('rewardChart');
        if (!ctx) return;
        
        // Destroy existing chart if it exists
        if (appState.charts.reward) {
            appState.charts.reward.destroy();
            appState.charts.reward = null;
        }
        
        appState.charts.reward = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Optimization Reward',
                    data: [],
                    borderColor: CONFIG.COLORS.success,
                    backgroundColor: `${CONFIG.COLORS.success}20`,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: CONFIG.COLORS.success,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: {
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: CONFIG.COLORS.dim,
                        borderColor: CONFIG.COLORS.success,
                        borderWidth: 1,
                        displayColors: false,
                        callbacks: {
                            label: (context) => {
                                return `Reward: ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        ticks: {
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 11
                            },
                            callback: (value) => value.toFixed(2)
                        },
                        grid: {
                            color: 'rgba(55, 65, 81, 0.3)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Reward',
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 12
                            }
                        }
                    },
                    x: {
                        ticks: {
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 11
                            },
                            maxRotation: 45,
                            minRotation: 0
                        },
                        grid: {
                            color: 'rgba(55, 65, 81, 0.3)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Step',
                            color: CONFIG.COLORS.dim,
                            font: {
                                size: 12
                            }
                        }
                    }
                },
                animation: {
                    duration: 300
                }
            }
        });
    }
    
    /**
     * Update loss chart with new data
     */
    static updateLossChart() {
        if (!appState.charts.loss) return;
        
        // Limit data points
        let data = appState.lossData.slice(-CONFIG.CHART_MAX_POINTS);
        
        // Update chart data
        appState.charts.loss.data.labels = data.map(d => `Epoch ${d.epoch}`);
        appState.charts.loss.data.datasets[0].data = data.map(d => d.loss);
        
        // Update with animation
        appState.charts.loss.update('active');
    }
    
    /**
     * Update reward chart with new data
     */
    static updateRewardChart() {
        if (!appState.charts.reward) return;
        
        // Limit data points
        let data = appState.rewardData.slice(-CONFIG.CHART_MAX_POINTS);
        
        // Update chart data
        appState.charts.reward.data.labels = data.map(d => `Step ${d.step}`);
        appState.charts.reward.data.datasets[0].data = data.map(d => d.reward);
        
        // Update with animation
        appState.charts.reward.update('active');
    }
    
    /**
     * Clear all charts
     */
    static clearCharts() {
        if (appState.charts.loss) {
            appState.charts.loss.data.labels = [];
            appState.charts.loss.data.datasets[0].data = [];
            appState.charts.loss.update();
        }
        
        if (appState.charts.reward) {
            appState.charts.reward.data.labels = [];
            appState.charts.reward.data.datasets[0].data = [];
            appState.charts.reward.update();
        }
    }
}

// ============================================================================
// Phase Banner Management
// ============================================================================

class PhaseBanner {
    constructor() {
        this.phases = [
            { id: 'planning', name: 'Planning', icon: '📋' },
            { id: 'expert_demonstration', name: 'Expert Demo', icon: '👨‍🏫' },
            { id: 'behavior_cloning', name: 'BC Training', icon: '🧠' },
            { id: 'optimization', name: 'Optimizing', icon: '⚙️' },
            { id: 'vision_refinement', name: 'Vision', icon: '👁️' },
            { id: 'code_generation', name: 'Codegen', icon: '💻' },
            { id: 'execution', name: 'Execution', icon: '🤖' }
        ];
        
        this.currentPhase = null;
        this.phaseTimes = {};
        this.phaseStartTimes = {};
    }
    
    /**
     * Update phase status in the banner
     */
    updatePhase(phaseName, status = 'active') {
        // Clean up phase name
        const cleanPhase = phaseName.toLowerCase().replace(/-/g, '_');
        
        // Update previous phase to completed if moving to new phase
        if (this.currentPhase && this.currentPhase !== cleanPhase && status === 'active') {
            this.setPhaseStatus(this.currentPhase, 'completed');
            this.recordPhaseTime(this.currentPhase);
        }
        
        // Set new phase status
        this.setPhaseStatus(cleanPhase, status);
        
        // Update current phase tracking
        if (status === 'active') {
            this.currentPhase = cleanPhase;
            this.phaseStartTimes[cleanPhase] = Date.now();
            this.updateCurrentPhaseDisplay(cleanPhase);
        }
        
        // Handle completion
        if (cleanPhase === 'completed') {
            this.handlePipelineComplete(true);
        } else if (cleanPhase === 'failed') {
            this.handlePipelineComplete(false);
        }
    }
    
    /**
     * Set visual status of a phase
     */
    setPhaseStatus(phaseId, status) {
        const phaseElement = document.querySelector(`[data-phase="${phaseId}"]`);
        if (!phaseElement) return;
        
        // Remove all status classes
        phaseElement.classList.remove('pending', 'active', 'completed', 'failed');
        
        // Add new status class
        phaseElement.classList.add(status);
        
        // Special handling for different statuses
        if (status === 'active') {
            // Add pulsing animation to icon
            const icon = phaseElement.querySelector('.phase-icon');
            if (icon) {
                icon.style.animation = 'pulsate 2s infinite';
            }
        } else if (status === 'completed') {
            // Remove animation from icon
            const icon = phaseElement.querySelector('.phase-icon');
            if (icon) {
                icon.style.animation = 'none';
            }
            // Update connector to show completion
            const connector = phaseElement.querySelector('.phase-connector');
            if (connector) {
                connector.style.background = 'var(--secondary-color)';
            }
        }
    }
    
    /**
     * Record time for a completed phase
     */
    recordPhaseTime(phaseId) {
        if (this.phaseStartTimes[phaseId]) {
            const duration = Date.now() - this.phaseStartTimes[phaseId];
            this.phaseTimes[phaseId] = duration;
            
            // Update time display
            const timeElement = document.getElementById(`${phaseId.replace('_', '-')}-time`);
            if (timeElement) {
                timeElement.textContent = this.formatDuration(duration);
            }
        }
    }
    
    /**
     * Update the current phase display
     */
    updateCurrentPhaseDisplay(phaseId) {
        const phase = this.phases.find(p => p.id === phaseId);
        if (!phase) return;
        
        const titleEl = document.getElementById('currentPhaseTitle');
        const subtitleEl = document.getElementById('currentPhaseSubtitle');
        
        if (titleEl) {
            titleEl.textContent = phase.name;
        }
        
        if (subtitleEl) {
            const messages = {
                'planning': 'Generating task plan...',
                'expert_demonstration': 'Collecting expert demonstrations...',
                'behavior_cloning': 'Training behavior cloning policy...',
                'optimization': 'Optimizing policy with reinforcement learning...',
                'vision_refinement': 'Processing vision feedback...',
                'code_generation': 'Generating executable code...',
                'execution': 'Executing on robot...'
            };
            subtitleEl.textContent = messages[phaseId] || 'Processing...';
        }
    }
    
    /**
     * Handle pipeline completion
     */
    handlePipelineComplete(success) {
        // Mark all phases as completed or failed
        this.phases.forEach(phase => {
            const element = document.querySelector(`[data-phase="${phase.id}"]`);
            if (element && !element.classList.contains('completed')) {
                if (success) {
                    element.classList.add('completed');
                } else {
                    element.classList.add('failed');
                }
            }
        });
        
        // Update current phase display
        const titleEl = document.getElementById('currentPhaseTitle');
        const subtitleEl = document.getElementById('currentPhaseSubtitle');
        
        if (titleEl) {
            titleEl.textContent = success ? 'Complete! ✅' : 'Failed ❌';
        }
        
        if (subtitleEl) {
            if (success) {
                const totalTime = Object.values(this.phaseTimes).reduce((a, b) => a + b, 0);
                subtitleEl.textContent = `Pipeline completed in ${this.formatDuration(totalTime)}`;
            } else {
                subtitleEl.textContent = 'Pipeline execution failed. Check console for details.';
            }
        }
    }
    
    /**
     * Reset all phases to pending
     */
    reset() {
        this.currentPhase = null;
        this.phaseTimes = {};
        this.phaseStartTimes = {};
        
        // Reset all phase elements
        this.phases.forEach(phase => {
            const element = document.querySelector(`[data-phase="${phase.id}"]`);
            if (element) {
                element.classList.remove('active', 'completed', 'failed');
                element.classList.add('pending');
                
                // Reset time display
                const timeEl = document.getElementById(`${phase.id.replace('_', '-')}-time`);
                if (timeEl) {
                    timeEl.textContent = '--';
                }
                
                // Reset connector
                const connector = element.querySelector('.phase-connector');
                if (connector) {
                    connector.style.background = 'var(--border-color)';
                }
            }
        });
        
        // Reset current phase display
        const titleEl = document.getElementById('currentPhaseTitle');
        const subtitleEl = document.getElementById('currentPhaseSubtitle');
        
        if (titleEl) {
            titleEl.textContent = 'Ready';
        }
        if (subtitleEl) {
            subtitleEl.textContent = 'Waiting for task execution...';
        }
    }
    
    /**
     * Format duration in ms to readable format
     */
    formatDuration(ms) {
        if (ms < 1000) {
            return `${ms}ms`;
        } else if (ms < 60000) {
            return `${(ms / 1000).toFixed(1)}s`;
        } else {
            const minutes = Math.floor(ms / 60000);
            const seconds = ((ms % 60000) / 1000).toFixed(0);
            return `${minutes}m ${seconds}s`;
        }
    }
}

// Global phase banner instance
const phaseBanner = new PhaseBanner();

// ============================================================================
// UI Management
// ============================================================================

class UI {
    /**
     * Update progress bar
     */
    static updateProgress(percent, text) {
        const fillEl = document.getElementById('progressFill');
        const textEl = document.getElementById('progressText');
        const rateEl = document.getElementById('completionRate');
        
        if (fillEl) {
            fillEl.style.width = `${percent}%`;
        }
        
        if (textEl) {
            textEl.textContent = text || `${Math.round(percent)}% complete`;
        }
        
        if (rateEl) {
            rateEl.textContent = Math.round(percent);
        }
    }
    
    /**
     * Update stage display
     */
    static updateStage(stage) {
        const stageEl = document.getElementById('currentStage');
        if (stageEl) {
            const formatted = stage.replace(/_/g, ' ').toUpperCase();
            stageEl.textContent = formatted;
        }
    }
    
    /**
     * Update metric display
     */
    static updateMetric(metric, value) {
        const elements = {
            bcLoss: 'bcLossValue',
            reward: 'rewardValue'
        };
        
        const el = document.getElementById(elements[metric]);
        if (el) {
            if (value === null || value === undefined) {
                el.textContent = '-';
            } else {
                const decimals = metric === 'bcLoss' ? 4 : 3;
                el.textContent = value.toFixed(decimals);
            }
        }
    }
    
    /**
     * Set status badge
     */
    static setStatus(status) {
        const badge = document.getElementById('statusBadge');
        if (badge) {
            badge.className = `status-badge ${status}`;
            
            const statusText = {
                'idle': 'Idle',
                'running': 'Running',
                'success': 'Success',
                'error': 'Error'
            };
            
            badge.textContent = statusText[status] || status;
        }
    }
    
    /**
     * Reset UI controls
     */
    static resetControls() {
        const btn = document.getElementById('executeBtn');
        if (btn) {
            btn.disabled = false;
            document.getElementById('btnIcon').textContent = '▶';
            document.getElementById('btnText').textContent = 'Execute';
        }
        
        this.setStatus('idle');
        appState.isExecuting = false;
    }
    
    /**
     * Show generated code
     */
    static showCode(code, codeLinks) {
        const panel = document.getElementById('codePanel');
        const codeEl = document.getElementById('generatedCode');
        
        if (panel && codeEl) {
            // Show first 50 lines
            const lines = code.split('\n').slice(0, 50);
            codeEl.textContent = lines.join('\n') + '\n\n# ... (truncated)';
            
            panel.style.display = 'block';
            
            // Setup download buttons
            if (codeLinks) {
                const downloadBtn = document.getElementById('downloadCode');
                const viewBtn = document.getElementById('viewCode');
                
                if (downloadBtn) {
                    downloadBtn.onclick = () => window.open(codeLinks.code_download_url, '_blank');
                }
                
                if (viewBtn) {
                    viewBtn.onclick = () => window.open(codeLinks.code_preview_url, '_blank');
                }
            }
        }
    }
}

// ============================================================================
// Timer Management
// ============================================================================

class Timer {
    static start() {
        this.stop();  // Clear any existing timer
        
        appState.executionStartTime = Date.now();
        
        appState.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - appState.executionStartTime) / 1000);
            const el = document.getElementById('timeElapsed');
            if (el) {
                el.textContent = elapsed;
            }
        }, 1000);
    }
    
    static stop() {
        if (appState.timerInterval) {
            clearInterval(appState.timerInterval);
            appState.timerInterval = null;
        }
    }
}

// ============================================================================
// Animation System
// ============================================================================

class Animation {
    static showSuccess() {
        const canvas = document.getElementById('simCanvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        let opacity = 0;
        let scale = 0.5;
        
        function draw() {
            // Draw regular simulation frame first
            Simulation.drawFrame(ctx, 0);
            
            // Draw success overlay
            ctx.save();
            ctx.globalAlpha = opacity;
            ctx.translate(canvas.width / 2, canvas.height / 2);
            ctx.scale(scale, scale);
            
            // Success circle
            ctx.strokeStyle = CONFIG.COLORS.success;
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.arc(0, 0, 40, 0, Math.PI * 2);
            ctx.stroke();
            
            // Checkmark
            ctx.beginPath();
            ctx.moveTo(-15, 0);
            ctx.lineTo(-5, 10);
            ctx.lineTo(15, -10);
            ctx.stroke();
            
            ctx.restore();
            
            opacity = Math.min(opacity + 0.05, 1);
            scale = Math.min(scale + 0.05, 1);
            
            if (opacity < 1 || scale < 1) {
                requestAnimationFrame(draw);
            }
        }
        
        draw();
    }
}

// ============================================================================
// Simulation Canvas
// ============================================================================

class Simulation {
    static initialize() {
        const canvas = document.getElementById('simCanvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // Start animation loop
        let frame = 0;
        const animate = () => {
            this.drawFrame(ctx, frame);
            frame = (frame + 1) % 360;
            appState.animationFrame = requestAnimationFrame(animate);
        };
        animate();
    }
    
    static drawFrame(ctx, frame) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        // Clear canvas
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);
        
        // Draw grid
        ctx.strokeStyle = 'rgba(55, 65, 81, 0.3)';
        ctx.lineWidth = 1;
        
        const gridSize = 20;
        for (let x = 0; x <= width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
        for (let y = 0; y <= height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw robot arm
        const centerX = width / 2;
        const centerY = height / 2;
        const armLength = 60;
        
        // Base
        ctx.fillStyle = '#4b5563';
        ctx.fillRect(centerX - 20, centerY + 40, 40, 20);
        
        // Animated angles
        const angle1 = Math.sin(frame * CONFIG.ANIMATION_SPEED) * 0.3;
        const angle2 = Math.cos(frame * CONFIG.ANIMATION_SPEED * 1.5) * 0.2;
        
        // First arm segment
        ctx.strokeStyle = CONFIG.COLORS.primary;
        ctx.lineWidth = 6;
        ctx.lineCap = 'round';
        
        const x1 = centerX + Math.cos(angle1 - Math.PI/2) * armLength;
        const y1 = centerY + Math.sin(angle1 - Math.PI/2) * armLength;
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        
        // Second arm segment
        const x2 = x1 + Math.cos(angle1 + angle2 - Math.PI/2) * armLength * 0.8;
        const y2 = y1 + Math.sin(angle1 + angle2 - Math.PI/2) * armLength * 0.8;
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        
        // Gripper
        ctx.strokeStyle = CONFIG.COLORS.success;
        ctx.lineWidth = 4;
        
        const gripperSize = 15;
        const gripperAngle = angle1 + angle2 - Math.PI/2;
        
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 + Math.cos(gripperAngle - 0.3) * gripperSize, 
                  y2 + Math.sin(gripperAngle - 0.3) * gripperSize);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 + Math.cos(gripperAngle + 0.3) * gripperSize,
                  y2 + Math.sin(gripperAngle + 0.3) * gripperSize);
        ctx.stroke();
        
        // Draw cube (object)
        const cubeX = centerX + 80;
        const cubeY = centerY + 20;
        const cubeSize = 25;
        
        ctx.fillStyle = CONFIG.COLORS.info;
        ctx.fillRect(cubeX - cubeSize/2, cubeY - cubeSize/2, cubeSize, cubeSize);
        
        ctx.strokeStyle = '#60a5fa';
        ctx.lineWidth = 2;
        ctx.strokeRect(cubeX - cubeSize/2, cubeY - cubeSize/2, cubeSize, cubeSize);
        
        // Draw target platform
        ctx.fillStyle = CONFIG.COLORS.danger;
        ctx.fillRect(centerX - 100, centerY + 50, 60, 10);
        
        // Status text
        ctx.fillStyle = CONFIG.COLORS.dim;
        ctx.font = '12px Monaco';
        ctx.fillText('Simulation Active', 10, 20);
        ctx.fillText(`Frame: ${frame}`, 10, height - 10);
        
        if (appState.isExecuting) {
            ctx.fillText('Executing...', width - 100, 20);
        }
    }
}

// ============================================================================
// Task Execution
// ============================================================================

class TaskExecutor {
    static async execute() {
        const btn = document.getElementById('executeBtn');
        const prompt = document.getElementById('promptBox').value.trim();
        
        if (!prompt) {
            Logger.log('Please enter a task description', 'error');
            return;
        }
        
        // Update UI state
        btn.disabled = true;
        document.getElementById('btnIcon').textContent = '⏸';
        document.getElementById('btnText').textContent = 'Running...';
        UI.setStatus('running');
        appState.isExecuting = true;
        
        // Clear previous data
        appState.reset();
        Logger.clear();
        MetricsManager.clear();
        phaseBanner.reset();
        
        // Start timer
        Timer.start();
        
        // Prepare request
        const request = {
            task_type: 'pick_and_place',
            task_description: prompt,
            use_vision: document.getElementById('useVision').checked,
            use_gpt_reward: document.getElementById('useGPT').checked,
            dry_run: document.getElementById('dryRun').checked,
            num_bc_epochs: parseInt(document.getElementById('bcEpochs').value),
            num_optimization_steps: parseInt(document.getElementById('optSteps').value),
            safety_checks: true
        };
        
        Logger.log(`Executing task: ${prompt}`, 'info');
        Logger.log(`Configuration: Vision=${request.use_vision}, GPT=${request.use_gpt_reward}, Dry Run=${request.dry_run}`, 'info');
        
        try {
            // Send execution request to REAL CogniForge endpoint
            const response = await fetch(`${CONFIG.API_BASE}/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            appState.currentRequestId = result.request_id;
            
            Logger.log(`Request ID: ${appState.currentRequestId}`, 'info');
            
            // Connect to SSE for real-time updates
            const sseManager = new SSEManager();
            sseManager.connect(appState.currentRequestId);
            
            // Handle initial result
            this.handleResult(result);
            
        } catch (error) {
            Logger.log(`Execution failed: ${error.message}`, 'error');
            console.error('Execution error:', error);
            UI.resetControls();
            Timer.stop();
        }
    }
    
    static handleResult(result) {
        // Update stats from summary
        if (result.summary) {
            const summary = result.summary;
            
            document.getElementById('completionRate').textContent = 
                Math.round(summary.success_rate * 100);
            
            if (summary.final_bc_loss !== null) {
                UI.updateMetric('bcLoss', summary.final_bc_loss);
            }
            
            if (summary.final_optimization_reward !== null) {
                UI.updateMetric('reward', summary.final_optimization_reward);
            }
            
            // Log summary
            Logger.log('=== Execution Summary ===', 'success');
            Logger.log(`Total time: ${summary.total_duration_seconds.toFixed(2)}s`, 'success');
            Logger.log(`Stages completed: ${summary.stages_completed_count}/${summary.stages_total_count}`, 'success');
        }
        
        // Show generated code if available
        if (result.generated_code) {
            UI.showCode(result.generated_code, result.code_links);
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize components
    ChartManager.initialize();
    Simulation.initialize();
    
    // Setup event listeners
    const executeBtn = document.getElementById('executeBtn');
    if (executeBtn) {
        executeBtn.addEventListener('click', () => TaskExecutor.execute());
    }
    
    const promptBox = document.getElementById('promptBox');
    if (promptBox) {
        promptBox.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                TaskExecutor.execute();
            }
        });
    }
    
    // Clear console button
    window.clearConsole = () => Logger.clear();
    
    // Log ready message
    Logger.log('CogniForge system ready', 'success');
    Logger.log('Enter a task description and click Execute to begin', 'info');
});

// ============================================================================
// Export for debugging
// ============================================================================

window.CogniForge = {
    state: appState,
    config: CONFIG,
    Logger,
    SSEManager,
    ChartManager,
    MetricsManager,
    PhaseBanner,
    phaseBanner,
    UI,
    Timer,
    Animation,
    Simulation,
    TaskExecutor
};
