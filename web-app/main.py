import os
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Spam Prediction Web App")

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Paths to model and vectorizer
MODEL_PATH = "spamprediction.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# Load the model and vectorizer
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    # If not found in current directory, try up one level (in case it's in the root)
    if os.path.exists(f"../{MODEL_PATH}") and os.path.exists(f"../{VECTORIZER_PATH}"):
        model = joblib.load(f"../{MODEL_PATH}")
        vectorizer = joblib.load(f"../{VECTORIZER_PATH}")
    else:
        # Fallback if I missed something in copy
        model = None
        vectorizer = None

# Define labels mapping
ID2LABEL = {0: 'Ham (Original)', 1: 'Spam'}

def preprocess_text(text: str):
    """Simple preprocessing as per the notebook."""
    text = str(text)
    # Matching the notebook's split + stopword check logic
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

def predict_spam(text: str):
    """Predict if the text is spam or not."""
    if model is None or vectorizer is None:
        return "Model not loaded"
        
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    
    try:
        # XGBoost trained on dense array in notebook
        text_dense = text_vectorized.toarray()
        prediction = int(model.predict(text_dense)[0])
    except Exception as e:
        # Fallback to direct prediction
        prediction = int(model.predict(text_vectorized)[0])
        
    return ID2LABEL.get(prediction, "Unknown")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Main page for the spam prediction app."""
    html_content = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Detection AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            body { 
                font-family: 'Plus Jakarta Sans', sans-serif; 
                background-color: #020617;
                background-image: 
                    radial-gradient(at 0% 0%, rgba(30, 20, 100, 0.3) 0, transparent 50%), 
                    radial-gradient(at 100% 100%, rgba(50, 20, 80, 0.3) 0, transparent 50%);
                color: #f8fafc;
                margin: 0;
                overflow-x: hidden;
            }
            .glass { 
                background: rgba(15, 23, 42, 0.6); 
                backdrop-filter: blur(16px); 
                border: 1px solid rgba(255, 255, 255, 0.08); 
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            .gradient-text { 
                background: linear-gradient(135deg, #60a5fa, #c084fc, #f472b6); 
                -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
            }
            .input-box {
                background: rgba(2, 6, 23, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.05);
                transition: all 0.2s ease;
            }
            .input-box:focus-within {
                border-color: #6366f1;
                box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
            }
            .predict-btn {
                background: linear-gradient(135deg, #6366f1, #a855f7);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .predict-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 15px -3px rgba(168, 85, 247, 0.4);
            }
            .predict-btn:active {
                transform: scale(0.98);
            }
            .spinner { 
                border: 2px solid rgba(255, 255, 255, 0.1); 
                border-left-color: white; 
                border-radius: 50%; 
                width: 20px; 
                height: 20px; 
                animation: spin 0.8s linear infinite; 
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            .fade-in { animation: fadeIn 0.6s ease-out forwards; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.2); border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.4); }
        </style>
    </head>
    <body class="min-h-screen flex items-center justify-center p-6 sm:p-12">
        <div class="max-w-xl w-full glass rounded-[2.5rem] p-8 sm:p-10 fade-in">
            <header class="text-center mb-10">
                <div class="inline-flex items-center justify-center p-3 mb-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>
                </div>
                <h1 class="text-4xl font-extrabold tracking-tight mb-2 gradient-text">Spam Detector</h1>
                <p class="text-slate-400 text-lg font-medium">Berdasarkan Notebook XGBClassifier</p>
            </header>

            <div class="space-y-6">
                <div>
                    <label class="block text-sm font-semibold text-slate-400 mb-2 px-1">Pesan yang ingin dianalisis</label>
                    <div class="input-box rounded-2xl p-4 overflow-hidden">
                        <textarea id="message" rows="4" 
                            class="w-full bg-transparent border-none outline-none text-slate-100 placeholder-slate-600 resize-none text-lg" 
                            placeholder="Ketik pesan Anda di sini..."></textarea>
                    </div>
                </div>
                
                <button onclick="analyzeMessage()" id="predict-btn" 
                    class="predict-btn w-full h-14 rounded-2xl text-white font-bold text-lg flex items-center justify-center gap-3">
                    <span id="btn-text">Check for Spam</span>
                    <div id="loader" class="spinner hidden"></div>
                </button>
            </div>

            <div id="result-container" class="mt-10 overflow-hidden transition-all duration-500 opacity-0 scale-95 h-0">
                <div id="result-box" class="rounded-2xl p-6 border-2 flex flex-col items-center gap-3">
                    <p class="text-sm font-bold uppercase tracking-[0.2em] opacity-60">Result Analysis</p>
                    <h2 id="prediction-text" class="text-4xl font-black">---</h2>
                </div>
            </div>

            <div class="mt-10 pt-8 border-t border-slate-800/60">
                <h3 class="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 px-1">Riwayat Analisis</h3>
                <div id="history-list" class="space-y-3 max-h-48 overflow-y-auto pr-1">
                    <p class="text-slate-600 italic text-sm text-center py-4">Belum ada riwayat analisis.</p>
                </div>
            </div>
        </div>

        <script>
            async function analyzeMessage() {
                const text = document.getElementById('message').value.trim();
                const btn = document.getElementById('predict-btn');
                const btnText = document.getElementById('btn-text');
                const loader = document.getElementById('loader');
                const container = document.getElementById('result-container');
                const resultBox = document.getElementById('result-box');
                const predictionText = document.getElementById('prediction-text');

                if (!text) {
                    alert('Mohon masukkan pesan terlebih dahulu!');
                    return;
                }

                // UI Loading State
                btn.disabled = true;
                btnText.textContent = "Analyzing...";
                loader.classList.remove('hidden');
                container.style.opacity = "0";
                container.style.height = "0";

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                        body: new URLSearchParams({ 'text': text })
                    });
                    
                    const data = await response.json();
                    
                    // Display results with animation
                    predictionText.textContent = data.prediction;
                    container.style.height = "auto";
                    container.classList.remove('opacity-0', 'scale-95');
                    container.classList.add('opacity-100', 'scale-100');
                    container.style.height = container.scrollHeight + "px";

                    if (data.prediction.includes('Spam')) {
                        resultBox.className = "rounded-2xl p-6 flex flex-col items-center gap-3 bg-red-500/10 border-red-500/30 text-red-400";
                    } else {
                        resultBox.className = "rounded-2xl p-6 flex flex-col items-center gap-3 bg-emerald-500/10 border-emerald-500/30 text-emerald-400";
                    }

                    addToHistory(text, data.prediction);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Terjadi kesalahan saat menghubungi server.');
                } finally {
                    btn.disabled = false;
                    btnText.textContent = "Check for Spam";
                    loader.classList.add('hidden');
                }
            }

            function addToHistory(text, result) {
                const historyList = document.getElementById('history-list');
                
                // Remove empty state message
                if (historyList.querySelector('p')) {
                    historyList.innerHTML = '';
                }

                const item = document.createElement('div');
                item.className = "glass rounded-xl p-4 flex justify-between items-center bg-slate-800/30 border-slate-700/50 animate-fade-in";
                
                const shortText = text.length > 35 ? text.substring(0, 35) + '...' : text;
                const isSpam = result.includes('Spam');
                
                item.innerHTML = `
                    <div class="flex flex-col">
                        <span class="text-slate-200 text-sm font-medium truncate w-40 sm:w-64">${shortText}</span>
                        <span class="text-[10px] text-slate-500 font-bold uppercase mt-1">Just now</span>
                    </div>
                    <span class="px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-wider ${isSpam ? 'bg-red-500/20 text-red-500' : 'bg-emerald-500/20 text-emerald-500'}">
                        ${result}
                    </span>
                `;
                
                historyList.prepend(item);
                if (historyList.children.length > 5) {
                    historyList.removeChild(historyList.lastChild);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(text: str = Form(...)):
    """API endpoint to predict spam."""
    prediction = predict_spam(text)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
