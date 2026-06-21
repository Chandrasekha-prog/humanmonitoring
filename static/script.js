// Global state
let featureSliders = {};

// Initialize features
function initializeFeatures() {
    const grid = document.getElementById('featureGrid');
    grid.innerHTML = '';
    
    // Show top 10 features for manual input
    const topFeatures = features.slice(0, 10);
    
    topFeatures.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'feature-item';
        
        const featureName = item.feature.length > 30 ? 
            item.feature.substring(0, 27) + '...' : 
            item.feature;
        
        div.innerHTML = `
            <label>
                <span title="${item.feature}">${featureName}</span>
                <span class="feature-value" id="value_${index}">0.00</span>
            </label>
            <input type="range" 
                   id="slider_${index}" 
                   min="-1" 
                   max="1" 
                   step="0.01" 
                   value="0">
        `;
        
        grid.appendChild(div);
        
        // Store slider reference
        const slider = document.getElementById(`slider_${index}`);
        featureSliders[item.feature] = slider;
        
        // Update value display on change
        slider.addEventListener('input', function() {
            const display = document.getElementById(`value_${index}`);
            display.textContent = parseFloat(this.value).toFixed(3);
        });
    });
}

// Initialize legend
function initializeLegend() {
    const grid = document.getElementById('legendGrid');
    const emojis = ['🚶', '🚶‍♂️', '🚶‍♀️', '🪑', '🧍', '🛏️'];
    
    activities.forEach((activity, index) => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const emoji = emojis[index % emojis.length] || '📱';
        item.textContent = `${emoji} ${activity}`;
        grid.appendChild(item);
    });
}

// Initialize tabs
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            document.getElementById(this.dataset.tab).classList.add('active');
        });
    });
}

// Initialize drag and drop
function initializeDragDrop() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('csvFile');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFileSelect(this.files[0]);
        }
    });
}

function handleFileSelect(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = fileInfo.querySelector('.file-name');
    
    fileName.textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    fileInfo.style.display = 'flex';
    window.uploadedFile = file;
}

// Initialize feature importance chart
function initializeFeatureImportance() {
    const container = document.getElementById('importanceChart');
    const topFeatures = features.slice(0, 20);
    
    let html = '';
    const maxImportance = topFeatures[0]?.importance || 1;
    
    topFeatures.forEach(item => {
        const width = (item.importance / maxImportance * 100);
        html += `
            <div class="chart-bar">
                <span class="chart-label" title="${item.feature}">${item.feature}</span>
                <div style="flex: 1;">
                    <div class="chart-bar-fill" style="width: ${width}%"></div>
                </div>
                <span class="chart-value">${(item.importance * 100).toFixed(2)}%</span>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Predict from CSV
async function predictCSV() {
    const file = window.uploadedFile;
    if (!file) {
        alert('Please select a CSV file first');
        return;
    }
    
    const container = document.getElementById('csvResults');
    const modelSelect = document.getElementById('csvModelSelect');
    container.innerHTML = '<div class="spinner"></div> Loading...';
    
    try {
        const content = await file.text();
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'csv',
                content: content,
                model: modelSelect.value
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="error">❌ ${data.error}</div>`;
            return;
        }
        
        displayCSVResults(data);
    } catch (error) {
        container.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

function displayCSVResults(data) {
    const container = document.getElementById('csvResults');
    
    let html = `<div class="result-card fade-in">
        <div class="result-header">
            <strong>📊 Predictions (${data.results.length} rows)</strong>
            <span class="model-used">🤖 ${data.model_used}</span>
        </div>
        <div style="overflow-x: auto;">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Row</th>
                        <th>Activity</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>`;
    
    data.results.forEach(result => {
        const confidenceColor = result.confidence > 80 ? '#10b981' : 
                               result.confidence > 60 ? '#f59e0b' : '#ef4444';
        html += `<tr>
            <td>${result.row}</td>
            <td><strong>${result.activity}</strong></td>
            <td style="color: ${confidenceColor}; font-weight: 600;">
                ${result.confidence.toFixed(1)}%
            </td>
        </tr>`;
    });
    
    html += `</tbody></table></div></div>`;
    container.innerHTML = html;
}

// Predict from manual input
async function predictManual() {
    const container = document.getElementById('manualResults');
    const modelSelect = document.getElementById('manualModelSelect');
    container.innerHTML = '<div class="spinner"></div> Predicting...';
    
    try {
        const featureValues = {};
        const sliderItems = document.querySelectorAll('#featureGrid .feature-item');
        
        sliderItems.forEach((item) => {
            const slider = item.querySelector('input[type="range"]');
            const label = item.querySelector('label span:first-child');
            let featureName = label.getAttribute('title') || label.textContent.trim();
            
            if (featureName.endsWith('...')) {
                const fullFeature = features.find(f => f.feature.startsWith(featureName.replace('...', '')));
                if (fullFeature) {
                    featureName = fullFeature.feature;
                }
            }
            
            featureValues[featureName] = parseFloat(slider.value);
        });
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'manual',
                features: featureValues,
                model: modelSelect.value
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="error">❌ ${data.error}</div>`;
            return;
        }
        
        displayManualResults(data);
    } catch (error) {
        container.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

function displayManualResults(data) {
    const container = document.getElementById('manualResults');
    
    let html = `<div class="result-card fade-in">
        <div class="result-header">
            <span class="result-activity">🎯 ${data.prediction}</span>
            <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                <span class="result-confidence">${data.confidence.toFixed(1)}%</span>
                <span class="model-used">🤖 ${data.model_used}</span>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${data.confidence}%">
                ${data.confidence.toFixed(1)}%
            </div>
        </div>
        <div style="margin-top: 16px;">
            <strong>🔝 Top 3 Predictions:</strong>
            <div style="margin-top: 8px;">`;
    
    data.top_predictions.forEach((pred, index) => {
        const colors = ['#10b981', '#3b82f6', '#8b5cf6'];
        html += `<div style="margin: 4px 0; display: flex; justify-content: space-between;">
            <span><span style="font-weight: 500;">${index + 1}.</span> ${pred.activity}</span>
            <span style="color: ${colors[index]}; font-weight: 600;">
                ${pred.probability.toFixed(1)}%
            </span>
        </div>`;
    });
    
    html += `</div></div></div>`;
    container.innerHTML = html;
}

// Predict samples
async function predictSample() {
    const container = document.getElementById('sampleResults');
    const modelSelect = document.getElementById('sampleModelSelect');
    container.innerHTML = '<div class="spinner"></div> Generating samples...';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'sample',
                model: modelSelect.value
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="error">❌ ${data.error}</div>`;
            return;
        }
        
        displaySampleResults(data);
    } catch (error) {
        container.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

function displaySampleResults(data) {
    const container = document.getElementById('sampleResults');
    
    let html = `<div class="fade-in">
        <div class="result-header">
            <strong>📊 Sample Predictions</strong>
            <span class="model-used">🤖 ${data.model_used}</span>
        </div>
        <div class="results-grid">`;
    
    data.results.forEach(result => {
        const isCorrect = result.true_activity === result.predicted_activity;
        html += `<div class="sample-card ${isCorrect ? 'correct' : 'incorrect'}">
            <div class="sample-id">Sample #${result.sample_id}</div>
            <div class="sample-activity">Predicted: ${result.predicted_activity}</div>
            <div class="sample-true">True: ${result.true_activity}</div>
            <div class="sample-confidence">Confidence: ${result.confidence.toFixed(1)}%</div>
            <div style="margin-top: 8px; font-size: 13px;">
                ${isCorrect ? '✅ Correct' : '❌ Incorrect'}
            </div>
        </div>`;
    });
    
    html += `</div></div>`;
    container.innerHTML = html;
}

// Reset sliders
function resetSliders() {
    document.querySelectorAll('#featureGrid input[type="range"]').forEach(slider => {
        slider.value = 0;
        slider.dispatchEvent(new Event('input'));
    });
}
// Update tab initialization to include real-time
function initializeTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            const tabId = this.dataset.tab;
            document.getElementById(tabId).classList.add('active');
            
            // If switching to real-time tab, check sensor status
            if (tabId === 'realtime') {
                checkSensorStatus();
            }
        });
    });
}

// Check sensor status
function checkSensorStatus() {
    const status = document.getElementById('sensorStatus');
    if (sensorActive) {
        status.innerHTML = '✅ Sensors active';
    } else {
        const hasSensors = 'Accelerometer' in window || 'Gyroscope' in window || 'DeviceOrientationEvent' in window;
        if (hasSensors) {
            status.innerHTML = '⚪ Sensors ready - click Start';
        } else {
            status.innerHTML = '⚠️ No sensors available';
        }
    }
}