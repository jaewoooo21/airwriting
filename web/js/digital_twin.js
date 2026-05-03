/**
 * Digital Twin Engine — Advanced 3D Visualization + System Monitor
 * ================================================================
 * 
 * 차세대 관성 지능 시스템의 전체 파이프라인을 실시간 시각화.
 * 
 * Panels:
 *   1. Foundation Model Status (LoRA adaptation progress)
 *   2. Duo Streamers State Machine (3-stage sparse recognition)
 *   3. FastKAN Network Visualization (learnable functions)
 *   4. Bio Kinematics View (ICOR trajectories + joint angles)
 *   5. Energy Efficiency Monitor (compute savings graph)
 *   6. Online TTA Feedback Loop (drift compensation)
 */

// ─── Digital Twin Dashboard Controller ───

class InertialIntelligenceDashboard {
    constructor(wsUrl) {
        this.ws = null;
        this.wsUrl = wsUrl || `ws://${location.hostname}:12347`;
        this.isConnected = false;
        
        // Pipeline state
        this.pipelineState = {
            foundationModel: { status: 'idle', loraProgress: 0, ttaUpdates: 0 },
            duoStreamers: { stage: 0, stageName: 'IDLE', savings: 0, accepted: 0 },
            fastKAN: { inferenceMs: 0, confidence: 0, predictedClass: -1 },
            bioKinematics: { jointAngles: {}, icorMCP: [0,0], icorPIP: [0,0] },
            energy: { totalFrames: 0, detectorCalls: 0, recognizerCalls: 0 },
        };
        
        // Chart data
        this.energyHistory = [];
        this.lossHistory = [];
        this.ttaHistory = [];
        
        this.maxHistory = 200;
    }
    
    // ─── WebSocket Connection ───
    
    connect() {
        try {
            this.ws = new WebSocket(this.wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateConnectionUI(true);
                console.log('[DigitalTwin] Connected');
            };
            
            this.ws.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data);
                    this.handleMessage(data);
                } catch(e) {}
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateConnectionUI(false);
                setTimeout(() => this.connect(), 3000);
            };
            
            this.ws.onerror = () => {
                this.isConnected = false;
            };
        } catch(e) {
            console.error('[DigitalTwin] Connection error:', e);
        }
    }
    
    handleMessage(data) {
        // ─── Live frame pipeline data (embedded in every sensor frame from main.py) ───
        if (data.pipeline) {
            const p = data.pipeline;
            
            // Bio Kinematics
            if (p.joint_angles) {
                this.pipelineState.bioKinematics = {
                    jointAngles: p.joint_angles,
                    icorMCP: p.icor_mcp || [0,0],
                    icorPIP: p.icor_pip || [0,0],
                    writingIntent: p.writing_intent || 0,
                };
                this._renderThrottle('biokin', () => this.renderBioKinematicsPanel(), 200);
            }
            
            // Duo Streamers efficiency
            if (p.duo_streamers && p.duo_streamers.total_frames) {
                const ds = p.duo_streamers;
                this.pipelineState.duoStreamers = {
                    stage: ds.recognizer_calls > 0 ? 2 : 1,
                    savings: ds.savings_pct || '0%',
                    accepted: ds.accepted || 0,
                    rejected: ds.rejected || 0,
                };
                this.pipelineState.energy = {
                    totalFrames: ds.total_frames,
                    detectorCalls: ds.detector_calls,
                    recognizerCalls: ds.recognizer_calls,
                    savings_pct: ds.savings_pct,
                };
                this._renderThrottle('streamer', () => this.renderStreamerPanel(), 500);
                this._renderThrottle('energy', () => this.renderEnergyPanel(), 500);
            }
            
            // Writing Intent -> Foundation Model status proxy
            this.pipelineState.foundationModel.status = 
                (p.writing_intent > 0.1) ? 'active' : 'idle';
            this._renderThrottle('pipeline', () => this.renderPipelinePanel(), 1000);
        }
        
        // ─── Typed messages (explicit pipeline_status, etc.) ───
        if (data.type === 'pipeline_status') {
            Object.assign(this.pipelineState, data.state);
            this.renderPipelinePanel();
        }
        if (data.type === 'streamer_state') {
            this.pipelineState.duoStreamers = data;
            this.renderStreamerPanel();
        }
        if (data.type === 'fastkan_result') {
            this.pipelineState.fastKAN = data;
            this.renderFastKANPanel();
        }
        if (data.type === 'bio_kinematics') {
            this.pipelineState.bioKinematics = data;
            this.renderBioKinematicsPanel();
        }
        if (data.type === 'energy_stats') {
            this.pipelineState.energy = data;
            this.renderEnergyPanel();
        }
    }
    
    // Throttle renders to avoid UI thrashing at 85Hz
    _renderThrottle(key, fn, intervalMs) {
        const now = Date.now();
        if (!this._throttleTimers) this._throttleTimers = {};
        if (!this._throttleTimers[key] || now - this._throttleTimers[key] >= intervalMs) {
            this._throttleTimers[key] = now;
            fn();
        }
    }
    
    // ─── UI Rendering ───
    
    updateConnectionUI(connected) {
        const dot = document.getElementById('dt-conn-dot');
        const text = document.getElementById('dt-conn-text');
        if (dot) {
            dot.className = `dt-dot ${connected ? 'connected' : ''}`;
        }
        if (text) {
            text.textContent = connected ? 'Online' : 'Offline';
        }
    }
    
    renderPipelinePanel() {
        const el = document.getElementById('dt-pipeline-status');
        if (!el) return;
        
        const st = this.pipelineState;
        el.innerHTML = `
            <div class="dt-pipeline-row">
                <span class="dt-label">Foundation Model</span>
                <span class="dt-value ${st.foundationModel.status === 'active' ? 'active' : ''}">${st.foundationModel.status}</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">LoRA Adaptation</span>
                <div class="dt-progress-bar">
                    <div class="dt-progress-fill" style="width:${st.foundationModel.loraProgress}%"></div>
                </div>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">TTA Updates</span>
                <span class="dt-value">${st.foundationModel.ttaUpdates}</span>
            </div>
        `;
    }
    
    renderStreamerPanel() {
        const el = document.getElementById('dt-streamer-status');
        if (!el) return;
        
        const ds = this.pipelineState.duoStreamers;
        const stageColors = ['#64748b', '#f59e0b', '#10b981', '#3b82f6'];
        const stageNames = ['IDLE', 'DETECTING', 'RECOGNIZING', 'VERIFYING'];
        
        el.innerHTML = `
            <div class="dt-streamer-stages">
                ${stageNames.map((name, i) => `
                    <div class="dt-stage ${ds.stage === i ? 'active' : ''}" 
                         style="--stage-color: ${stageColors[i]}">
                        <div class="dt-stage-indicator"></div>
                        <span>${name}</span>
                    </div>
                `).join('')}
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">Compute Savings</span>
                <span class="dt-value highlight">${ds.savings || 0}%</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">Accepted / Rejected</span>
                <span class="dt-value">${ds.accepted || 0} / ${ds.rejected || 0}</span>
            </div>
        `;
    }
    
    renderFastKANPanel() {
        const el = document.getElementById('dt-fastkan-status');
        if (!el) return;
        
        const fk = this.pipelineState.fastKAN;
        
        el.innerHTML = `
            <div class="dt-pipeline-row">
                <span class="dt-label">Inference</span>
                <span class="dt-value">${fk.inferenceMs?.toFixed(2) || '0.00'}ms</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">Confidence</span>
                <span class="dt-value">${((fk.confidence || 0) * 100).toFixed(1)}%</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">Memory</span>
                <span class="dt-value">35 KB</span>
            </div>
            <div class="dt-kan-equation">
                f(x) = &Sigma; &Phi;<sub>q</sub>( &Sigma; &phi;<sub>q,p</sub>(x<sub>p</sub>) )
            </div>
        `;
    }
    
    renderBioKinematicsPanel() {
        const el = document.getElementById('dt-biokin-status');
        if (!el) return;
        
        const bk = this.pipelineState.bioKinematics;
        const angles = bk.jointAngles || {};
        
        el.innerHTML = `
            <div class="dt-joint-grid">
                ${Object.entries(angles).map(([name, val]) => `
                    <div class="dt-joint-item">
                        <span class="dt-joint-name">${name.replace('_', ' ')}</span>
                        <span class="dt-joint-value">${(val * 180 / Math.PI).toFixed(1)}&deg;</span>
                    </div>
                `).join('')}
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">ICOR MCP</span>
                <span class="dt-value">[${(bk.icorMCP || [0,0]).map(v => (v*1000).toFixed(1)).join(', ')}] mm</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">ICOR PIP</span>
                <span class="dt-value">[${(bk.icorPIP || [0,0]).map(v => (v*1000).toFixed(1)).join(', ')}] mm</span>
            </div>
        `;
    }
    
    renderEnergyPanel() {
        const el = document.getElementById('dt-energy-status');
        if (!el) return;
        
        const en = this.pipelineState.energy;
        el.innerHTML = `
            <div class="dt-pipeline-row">
                <span class="dt-label">Total Frames</span>
                <span class="dt-value">${en.totalFrames || 0}</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">Detector Only</span>
                <span class="dt-value">${en.detectorCalls || 0}</span>
            </div>
            <div class="dt-pipeline-row">
                <span class="dt-label">Full Inference</span>
                <span class="dt-value">${en.recognizerCalls || 0}</span>
            </div>
            <div class="dt-energy-bar">
                <div class="dt-energy-saved" style="width:${Math.min(parseFloat(en.savings_pct || 0), 100)}%">
                    ${en.savings_pct || '0%'} saved
                </div>
            </div>
        `;
    }
    
    // ─── Create Dashboard HTML ───
    
    static createDashboardHTML() {
        return `
        <div id="dt-dashboard" class="dt-dashboard">
            <div class="dt-header">
                <h2>Inertial Intelligence Dashboard</h2>
                <div class="dt-conn">
                    <div class="dt-dot" id="dt-conn-dot"></div>
                    <span id="dt-conn-text">Offline</span>
                </div>
            </div>
            <div class="dt-grid">
                <div class="dt-card">
                    <h3>Foundation Model</h3>
                    <div id="dt-pipeline-status" class="dt-card-body">
                        <div class="dt-loading">Awaiting data...</div>
                    </div>
                </div>
                <div class="dt-card">
                    <h3>Duo Streamers</h3>
                    <div id="dt-streamer-status" class="dt-card-body">
                        <div class="dt-loading">Awaiting data...</div>
                    </div>
                </div>
                <div class="dt-card">
                    <h3>FastKAN Inference</h3>
                    <div id="dt-fastkan-status" class="dt-card-body">
                        <div class="dt-loading">Awaiting data...</div>
                    </div>
                </div>
                <div class="dt-card">
                    <h3>Bio Kinematics</h3>
                    <div id="dt-biokin-status" class="dt-card-body">
                        <div class="dt-loading">Awaiting data...</div>
                    </div>
                </div>
                <div class="dt-card dt-wide">
                    <h3>Energy Efficiency</h3>
                    <div id="dt-energy-status" class="dt-card-body">
                        <div class="dt-loading">Awaiting data...</div>
                    </div>
                </div>
            </div>
        </div>`;
    }
    
    static createDashboardCSS() {
        return `
        .dt-dashboard {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(20px);
            z-index: 1000;
            display: flex; flex-direction: column;
            font-family: 'Inter', -apple-system, sans-serif;
            color: #e2e8f0;
            overflow-y: auto;
        }
        .dt-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 20px 30px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .dt-header h2 {
            font-size: 18px; font-weight: 700;
            background: linear-gradient(135deg, #818cf8, #34d399);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: 1px;
        }
        .dt-conn { display: flex; align-items: center; gap: 8px; }
        .dt-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #ef4444; transition: all 0.3s;
        }
        .dt-dot.connected { background: #10b981; box-shadow: 0 0 8px #10b98166; }
        .dt-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px; padding: 20px 30px;
        }
        .dt-card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px; overflow: hidden;
        }
        .dt-card.dt-wide { grid-column: span 2; }
        .dt-card h3 {
            font-size: 11px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 1.5px;
            color: #94a3b8; padding: 14px 18px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .dt-card-body { padding: 14px 18px; }
        .dt-pipeline-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; font-size: 12px;
        }
        .dt-label { color: #94a3b8; }
        .dt-value { color: #e2e8f0; font-weight: 600; font-family: 'SF Mono', monospace; }
        .dt-value.active { color: #34d399; }
        .dt-value.highlight { color: #818cf8; font-size: 14px; }
        .dt-progress-bar {
            width: 100px; height: 4px; background: rgba(255,255,255,0.08);
            border-radius: 2px; overflow: hidden;
        }
        .dt-progress-fill {
            height: 100%; background: linear-gradient(90deg, #818cf8, #34d399);
            border-radius: 2px; transition: width 0.3s;
        }
        .dt-streamer-stages {
            display: flex; gap: 8px; margin-bottom: 12px;
        }
        .dt-stage {
            flex: 1; text-align: center; padding: 8px 4px;
            border-radius: 8px; font-size: 9px; font-weight: 600;
            letter-spacing: 0.5px; text-transform: uppercase;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.04);
            opacity: 0.4; transition: all 0.3s;
        }
        .dt-stage.active {
            opacity: 1; background: rgba(99, 102, 241, 0.15);
            border-color: var(--stage-color);
            box-shadow: 0 0 12px rgba(99, 102, 241, 0.2);
        }
        .dt-stage-indicator {
            width: 6px; height: 6px; border-radius: 50%;
            background: var(--stage-color); margin: 0 auto 4px;
        }
        .dt-kan-equation {
            text-align: center; padding: 10px;
            font-size: 14px; color: #818cf8;
            font-style: italic; margin-top: 8px;
            background: rgba(99, 102, 241, 0.05);
            border-radius: 8px;
        }
        .dt-joint-grid {
            display: grid; grid-template-columns: repeat(2, 1fr);
            gap: 6px; margin-bottom: 10px;
        }
        .dt-joint-item {
            display: flex; justify-content: space-between;
            padding: 4px 8px; background: rgba(255,255,255,0.02);
            border-radius: 6px; font-size: 11px;
        }
        .dt-joint-name { color: #94a3b8; text-transform: capitalize; }
        .dt-joint-value { color: #34d399; font-family: 'SF Mono', monospace; font-weight: 600; }
        .dt-energy-bar {
            height: 24px; background: rgba(255,255,255,0.05);
            border-radius: 6px; overflow: hidden; margin-top: 10px;
        }
        .dt-energy-saved {
            height: 100%; background: linear-gradient(90deg, #10b981, #34d399);
            border-radius: 6px; display: flex; align-items: center;
            justify-content: center; font-size: 11px; font-weight: 600;
            color: #0f172a; transition: width 0.5s;
        }
        .dt-loading {
            text-align: center; padding: 20px;
            color: #475569; font-size: 12px;
            font-style: italic;
        }
        `;
    }
}

// ─── Auto-init when loaded ───
if (typeof window !== 'undefined') {
    window.InertialIntelligenceDashboard = InertialIntelligenceDashboard;
}
