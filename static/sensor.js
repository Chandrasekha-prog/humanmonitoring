// Sensor handling
let sensorInterval = null;
let sensorActive = false;
let lastPredictionTime = 0;
const PREDICTION_INTERVAL = 1000; // Predict every 1 second

// Current sensor data
let currentAccel = { x: 0, y: 0, z: 0 };
let currentGyro = { x: 0, y: 0, z: 0 };
let currentOrientation = { alpha: 0, beta: 0, gamma: 0 };

// Start sensors
function startSensor() {
    if (sensorActive) return;
    
    const status = document.getElementById('sensorStatus');
    status.innerHTML = '🔄 Connecting to sensors...';
    
    // Check if sensors are supported
    const hasAccel = 'Accelerometer' in window;
    const hasGyro = 'Gyroscope' in window;
    const hasOrientation = 'DeviceOrientationEvent' in window;
    
    if (!hasAccel && !hasGyro && !hasOrientation) {
        status.innerHTML = '❌ No sensors available on this device';
        alert('Your device does not support required sensors. Please use a mobile device with sensors.');
        return;
    }
    
    // Use Generic Sensor API if available (Chrome Android)
    if (hasAccel && window.Accelerometer) {
        try {
            const accelerometer = new Accelerometer({ frequency: 60 });
            accelerometer.addEventListener('reading', () => {
                currentAccel.x = accelerometer.x || 0;
                currentAccel.y = accelerometer.y || 0;
                currentAccel.z = accelerometer.z || 0;
                updateSensorDisplay();
            });
            accelerometer.start();
        } catch (e) {
            console.warn('Accelerometer API failed:', e);
        }
    }
    
    if (hasGyro && window.Gyroscope) {
        try {
            const gyroscope = new Gyroscope({ frequency: 60 });
            gyroscope.addEventListener('reading', () => {
                currentGyro.x = gyroscope.x || 0;
                currentGyro.y = gyroscope.y || 0;
                currentGyro.z = gyroscope.z || 0;
                updateSensorDisplay();
            });
            gyroscope.start();
        } catch (e) {
            console.warn('Gyroscope API failed:', e);
        }
    }
    
    // Use DeviceOrientation API as fallback (works on most mobile browsers)
    if (hasOrientation) {
        window.addEventListener('deviceorientation', (event) => {
            currentOrientation.alpha = event.alpha || 0;
            currentOrientation.beta = event.beta || 0;
            currentOrientation.gamma = event.gamma || 0;
            updateSensorDisplay();
        });
        
        // Also use devicemotion for accelerometer if Generic API not available
        window.addEventListener('devicemotion', (event) => {
            const accel = event.accelerationIncludingGravity || event.acceleration;
            if (accel) {
                currentAccel.x = accel.x || 0;
                currentAccel.y = accel.y || 0;
                currentAccel.z = accel.z || 0;
                updateSensorDisplay();
            }
        });
    }
    
    // Start prediction interval
    sensorInterval = setInterval(() => {
        if (sensorActive) {
            predictFromSensor();
        }
    }, PREDICTION_INTERVAL);
    
    sensorActive = true;
    
    document.getElementById('startBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'inline-block';
    status.innerHTML = '✅ Sensors active - predicting...';
    
    // Trigger first prediction
    setTimeout(predictFromSensor, 500);
}

// Stop sensors
function stopSensor() {
    sensorActive = false;
    if (sensorInterval) {
        clearInterval(sensorInterval);
        sensorInterval = null;
    }
    
    document.getElementById('startBtn').style.display = 'inline-block';
    document.getElementById('stopBtn').style.display = 'none';
    document.getElementById('sensorStatus').innerHTML = '⏹️ Stopped';
}

// Update sensor display
function updateSensorDisplay() {

    console.log("Accelerometer:", currentAccel);
    console.log("Gyroscope:", currentGyro);
    console.log("Orientation:", currentOrientation);

    document.getElementById('accelData').innerHTML =
        `X: ${currentAccel.x.toFixed(3)}<br>Y: ${currentAccel.y.toFixed(3)}<br>Z: ${currentAccel.z.toFixed(3)}`;

    document.getElementById('gyroData').innerHTML =
        `X: ${currentGyro.x.toFixed(3)}<br>Y: ${currentGyro.y.toFixed(3)}<br>Z: ${currentGyro.z.toFixed(3)}`;

    document.getElementById('orientationData').innerHTML =
        `α: ${currentOrientation.alpha.toFixed(2)}<br>β: ${currentOrientation.beta.toFixed(2)}<br>γ: ${currentOrientation.gamma.toFixed(2)}`;
}
// Predict from current sensor data
async function predictFromSensor() {
    const modelSelect = document.getElementById('sensorModelSelect');
    const container = document.getElementById('realtimeResults');
    
    try {
        // Combine sensor data into features
        const features = {
            'tBodyAcc-mean()-X': currentAccel.x,
            'tBodyAcc-mean()-Y': currentAccel.y,
            'tBodyAcc-mean()-Z': currentAccel.z,
            'tBodyGyro-mean()-X': currentGyro.x,
            'tBodyGyro-mean()-Y': currentGyro.y,
            'tBodyGyro-mean()-Z': currentGyro.z,
            'angle(X,gravityMean)': currentOrientation.beta || 0,
            'angle(Y,gravityMean)': currentOrientation.gamma || 0
        };
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'realtime',
                features: features,
                model: modelSelect.value
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="error">❌ ${data.error}</div>`;
            return;
        }
        
        displayRealtimeResult(data);
    } catch (error) {
        console.error('Prediction error:', error);
        container.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
}

// Display real-time prediction
function displayRealtimeResult(data) {
    const container = document.getElementById('realtimeResults');
    
    const timestamp = new Date().toLocaleTimeString();
    
    let html = `<div class="result-card fade-in">
        <div class="result-header">
            <div>
                <span class="result-activity">🎯 ${data.prediction}</span>
                <span style="font-size: 14px; color: #6b7280; margin-left: 10px;">${timestamp}</span>
            </div>
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
        <div style="margin-top: 12px; display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;">`;
    
    data.top_predictions.forEach((pred, index) => {
        const colors = ['#10b981', '#3b82f6', '#8b5cf6'];
        html += `<div style="text-align: center;">
            <div style="font-size: 13px; color: #6b7280;">#${index + 1}</div>
            <div style="font-weight: 600; color: ${colors[index]};">${pred.activity}</div>
            <div style="font-size: 13px; color: ${colors[index]};">${pred.probability.toFixed(1)}%</div>
        </div>`;
    });
    
    html += `</div></div>`;
    container.innerHTML = html;
}

// Check sensor support on page load
document.addEventListener('DOMContentLoaded', function() {
    // Show sensor status
    const status = document.getElementById('sensorStatus');
    const hasAccel = 'Accelerometer' in window;
    const hasGyro = 'Gyroscope' in window;
    const hasOrientation = 'DeviceOrientationEvent' in window;
    
    if (hasAccel || hasGyro || hasOrientation) {
        status.innerHTML = '⚪ Ready - click Start';
    } else {
        status.innerHTML = '⚠️ No sensors detected';
    }
});