const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3000;

// Enable CORS for all routes
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, './');
    },
    filename: function (req, file, cb) {
        const timestamp = Date.now();
        const filename = `temp_eye_image_${timestamp}.jpg`;
        cb(null, filename);
    }
});

const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    }
});

// Serve the main HTML page
app.get('/', (req, res) => {
    console.log('EyeNemia Request: GET /');
    res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EyeNemia - Anemia Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .upload-area {
                border: 3px dashed rgba(255, 255, 255, 0.5);
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: rgba(255, 255, 255, 0.8);
                background: rgba(255, 255, 255, 0.1);
            }
            input[type="file"] {
                display: none;
            }
            .upload-btn {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px 0 rgba(255, 107, 107, 0.3);
            }
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px 0 rgba(255, 107, 107, 0.4);
            }
            .analyze-btn {
                background: linear-gradient(45deg, #4ecdc4, #44a08d);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                margin-top: 20px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px 0 rgba(78, 205, 196, 0.3);
            }
            .analyze-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px 0 rgba(78, 205, 196, 0.4);
            }
            .analyze-btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(5px);
            }
            .loading {
                text-align: center;
                font-size: 18px;
                color: #4ecdc4;
            }
            .preview-image {
                max-width: 300px;
                max-height: 300px;
                border-radius: 10px;
                margin: 20px auto;
                display: block;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }
            .health-status {
                font-size: 1.5em;
                font-weight: bold;
                margin: 15px 0;
                text-align: center;
            }
            .risk-high { color: #ff6b6b; }
            .risk-medium { color: #ffa726; }
            .risk-low { color: #66bb6a; }
            .details {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 20px 0;
            }
            .detail-item {
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .detail-label {
                font-size: 0.9em;
                opacity: 0.8;
                margin-bottom: 5px;
            }
            .detail-value {
                font-size: 1.2em;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>👁️ EyeNemia</h1>
            <p style="text-align: center; font-size: 1.2em; margin-bottom: 30px;">
                AI-Powered Anemia Detection from Eye Images
            </p>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p style="font-size: 1.1em; margin-bottom: 15px;">📸 Click to upload an eye image</p>
                <button class="upload-btn" type="button">Choose Image</button>
                <input type="file" id="fileInput" accept="image/*" onchange="previewImage(this)">
            </div>
            
            <div id="preview" style="display: none;">
                <img id="previewImg" class="preview-image" alt="Preview">
                <div style="text-align: center;">
                    <button class="analyze-btn" onclick="analyzeImage()">🔬 Analyze for Anemia</button>
                </div>
            </div>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>

        <script>
            let selectedFile = null;

            function previewImage(input) {
                if (input.files && input.files[0]) {
                    selectedFile = input.files[0];
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        document.getElementById('previewImg').src = e.target.result;
                        document.getElementById('preview').style.display = 'block';
                        document.getElementById('result').style.display = 'none';
                    }
                    
                    reader.readAsDataURL(input.files[0]);
                }
            }

            async function analyzeImage() {
                if (!selectedFile) {
                    alert('Please select an image first');
                    return;
                }

                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<div class="loading">🔬 Analyzing image... Please wait...</div>';

                const formData = new FormData();
                formData.append('image', selectedFile);

                try {
                    const response = await fetch('/api/analyze-image', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await response.json();
                    
                    if (result.success) {
                        displayResult(result);
                    } else {
                        resultDiv.innerHTML = '<div style="color: #ff6b6b;">❌ Error: ' + (result.error || 'Analysis failed') + '</div>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div style="color: #ff6b6b;">❌ Error: ' + error.message + '</div>';
                }
            }

            function displayResult(result) {
                const riskClass = result.riskLevel === 'High' || result.riskLevel === 'Critical' ? 'risk-high' : 
                                 result.riskLevel === 'Medium' ? 'risk-medium' : 'risk-low';
                
                const resultHTML = \`
                    <div class="health-status \${riskClass}">
                        \${result.healthStatus}
                    </div>
                    
                    <div class="details">
                        <div class="detail-item">
                            <div class="detail-label">Confidence</div>
                            <div class="detail-value">\${result.confidence}%</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Risk Level</div>
                            <div class="detail-value \${riskClass}">\${result.riskLevel}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Hemoglobin Est.</div>
                            <div class="detail-value">\${result.hemoglobinLevel} g/dL</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Classification</div>
                            <div class="detail-value">\${result.anemiaClassification}</div>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <strong>💡 Recommendations:</strong><br>
                        \${result.recommendations}
                    </div>
                    
                    <div style="font-size: 0.9em; opacity: 0.8; text-align: center; margin-top: 20px;">
                        Algorithm: \${result.algorithm}
                    </div>
                \`;
                
                document.getElementById('result').innerHTML = resultHTML;
            }
        </script>
    </body>
    </html>
    `);
});

// API endpoint for image analysis
app.post('/api/analyze-image', upload.single('image'), (req, res) => {
    console.log('EyeNemia Request: POST /api/analyze-image');
    
    if (!req.file) {
        return res.status(400).json({ error: 'No image file provided', success: false });
    }

    const imagePath = req.file.path;
    const imageSize = req.file.size;
    
    console.log(\`📁 Saved image: \${req.file.filename}, Size: \${imageSize} bytes\`);
    console.log(\`🔬 Starting WORKING anemia analysis for: \${req.file.filename}\`);

    // Call the working Python anemia detector
    const pythonProcess = spawn('python', ['working_anemia_detector.py', imagePath]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
        pythonOutput += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        pythonError += data.toString();
        console.log(\`🐍 Python: \${data.toString().trim()}\`);
    });

    pythonProcess.on('close', (code) => {
        // Clean up the temporary file
        fs.unlink(imagePath, (err) => {
            if (err) console.log(\`⚠️ Could not delete temp file: \${err.message}\`);
        });

        if (code !== 0) {
            console.log(\`❌ Python process exited with code \${code}\`);
            console.log(\`❌ Python error: \${pythonError}\`);
            return res.status(500).json({ 
                error: 'Analysis failed', 
                details: pythonError,
                success: false 
            });
        }

        try {
            const result = JSON.parse(pythonOutput.trim());
            
            // Add timestamp
            result.timestamp = new Date().toISOString();
            
            console.log(\`🔍 Working anemia analysis completed: \${JSON.stringify(result, null, 2)}\`);
            
            res.json(result);
        } catch (error) {
            console.log(\`❌ Error parsing Python output: \${error.message}\`);
            console.log(\`📄 Python output: \${pythonOutput}\`);
            res.status(500).json({ 
                error: 'Failed to parse analysis result', 
                details: error.message,
                success: false 
            });
        }
    });
});

// Start the server
app.listen(port, () => {
    console.log(\`🚀 EyeNemia Server running on port \${port}\`);
    console.log(\`📱 Open http://localhost:\${port} in your browser\`);
    console.log(\`👁️  EyeNemia - Your Eye Health Companion\`);
});
