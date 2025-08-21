# D3-DNA FastAPI Backend

GPU-accelerated DNA sequence generation API with web interface for the D3-DNA discrete diffusion model.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd api/
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# From project root
python run_api.py

# Or with custom options
python run_api.py --host 0.0.0.0 --port 2025
```

### 3. Access the Interface

- **Web Interface**: http://localhost:2025
- **API Docs**: http://localhost:2025/docs  
- **Health Check**: http://localhost:2025/health

### 4. SSH Tunneling (for remote GPU servers)

```bash
# On your local machine
ssh -N -L 2025:localhost:2025 user@gpu-server

# Then open http://localhost:2025 in your browser
```

## 📡 API Endpoints

### Generation
- `POST /generate` - Generate DNA sequences with full visualization data
- `GET /datasets/info` - Get available datasets and parameters
- `GET /models/status` - Check loaded model status
- `GET /health` - Health check and system status

### Frontend
- `GET /` - Web interface for interactive generation

## 🧬 Generation Features

### Core Generation Options
- **num_samples** (1-1000): Number of sequences to generate
- **steps** (1-500): Diffusion sampling steps
- **architecture**: "transformer" or "convolutional"
- **dataset**: "deepstarr" (extensible to other datasets)

### DeepSTARR-Specific Options  
- **dev_activity**: Dev enhancer activity conditioning
- **hk_activity**: HK enhancer activity conditioning
- **unconditional**: Generate without conditioning labels

### Evaluation Options
- **include_oracle**: Include oracle model predictions
- **include_sp_mse**: Compute SP-MSE against original data
- **specific_indices**: Target specific dataset samples (e.g., "11,12,40")
- **max_samples**: Limit dataset size for evaluation

### Visualization Options
- **include_visualization**: Capture step-by-step generation process
- **save_intermediate_steps**: Save all intermediate diffusion steps

## 📊 Response Format

The API returns complete visualization data in JSON format matching the H5→JSON structure:

```json
{
  "final_sequences": ["ATCGCTA...", "GCATAG..."],
  "generation_time": 12.34,
  "metadata": {
    "dataset": "deepstarr",
    "num_samples": 100,
    "sequence_length": 249,
    "total_steps": 50,
    "architecture": "transformer",
    "original_samples": [[0,1,2,3,...], ...],
    "ground_truth_labels": [[dev, hk], ...],
    "dataset_indices": [11, 12, 40, ...]
  },
  "steps": [
    {
      "step": 0,
      "timestep": 1.0,
      "noise_level": 20.0,
      "sequences": [[0,1,2,3,...], ...],
      "score_matrices": [[[...]], ...],
      "prob_matrices": [[[...]], ...],
      "oracle_mses": [0.123, 0.456, ...],
      "oracle_predictions": [[dev, hk], ...]
    }
  ],
  "sp_mse": 0.123
}
```

## 🧪 Testing

```bash
# Test the API
python api/test_api.py

# Manual test with curl
curl -X POST "http://localhost:2025/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "num_samples": 5,
    "steps": 20,
    "dev_activity": 2.0,
    "hk_activity": 1.5
  }'
```

## ⚙️ Configuration

### Environment Variables
- `API_HOST`: Server host (default: "localhost")
- `API_PORT`: Server port (default: 2025)
- `GPU_MEMORY_FRACTION`: GPU memory usage fraction (default: 0.8)
- `MAX_SAMPLES_PER_REQUEST`: Maximum samples per request (default: 1000)
- `CORS_ORIGINS`: CORS allowed origins (default: "*")

### Model Paths
Models and data are automatically loaded from config files in:
- `model_zoo/deepstarr/configs/transformer.yaml`
- `model_zoo/deepstarr/configs/convolutional.yaml`

## 🖥️ Web Interface

The simple web interface provides:

### Input Controls
- Sliders for samples and steps
- Dropdowns for architecture and dataset
- Checkboxes for evaluation and visualization options
- Number inputs for DeepSTARR conditioning

### Output Display
- **Generation Summary**: Metadata and timing
- **Sequences Preview**: First 3 generated DNA sequences
- **Step Data Preview**: First 2 diffusion steps
- **Oracle Predictions**: First 3 oracle predictions (if enabled)
- **Complete JSON**: Full API response (toggleable)

## 🔧 Architecture

```
api/
├── main.py              # FastAPI application with endpoints
├── models.py            # Pydantic request/response schemas
├── config.py            # Configuration management
├── services/
│   ├── model_loader.py  # GPU model loading and caching
│   ├── generation.py    # Core generation logic
│   └── visualization.py # Data formatting (H5→JSON)
└── static/
    ├── index.html       # Web interface
    ├── script.js        # Frontend JavaScript
    └── style.css        # Styling
```

## 🚀 GPU Acceleration

- Automatic GPU detection and utilization
- Model pre-loading for fast inference
- Memory management for large requests
- Batch processing optimization

## 🔍 Monitoring

- Health checks with GPU status
- Model loading status tracking
- Memory usage monitoring
- Generation timing metrics

## 🛠️ Development

```bash
# Run with auto-reload for development
python run_api.py --reload

# Format code
black api/
isort api/

# Type checking
mypy api/
```