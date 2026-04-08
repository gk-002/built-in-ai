# Supply Chain Inventory Management — OpenEnv Submission

**Author:** Ganesh Kendre  
**Event:** OpenEnv Hackathon — Round 1  
**Track:** Solo Warrior  

---

## 🏭 Environment Overview

A real-world simulation of **multi-warehouse supply chain inventory management**. An AI agent acts as a logistics operations manager, making restocking decisions across three warehouses to fulfill customer orders while minimizing costs.

This is NOT a toy or game — it mirrors the daily decisions made by supply chain teams at companies like Amazon, Flipkart, and Walmart.

---

## 🎯 The Task

The agent must:
1. **Monitor** inventory levels across 3 warehouses and 3 product categories
2. **Anticipate** demand using forecasts and pending order data
3. **Order restocks** proactively (accounting for product lead times: 1–3 steps)
4. **Fulfill** customer orders from available stock
5. **Minimize** holding costs and stockout penalties

---

## 🗂️ Project Structure

```
.
├── environment.py     # Core OpenEnv environment (step/reset/state API)
├── inference.py       # Baseline LLM inference script
├── app.py             # FastAPI server for HuggingFace Spaces
├── openenv.yaml       # OpenEnv spec file
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container build file
└── README.md          # This file
```

---

## 🔌 Action Space

```json
{
  "restock_orders": {
    "wh_north": {"prod_A": 50, "prod_B": 20, "prod_C": 0},
    "wh_south": {"prod_A": 0,  "prod_B": 15, "prod_C": 30},
    "wh_east":  {"prod_A": 10, "prod_B": 0,  "prod_C": 20}
  }
}
```

- **Warehouses:** `wh_north` (cap 500), `wh_south` (cap 400), `wh_east` (cap 300)
- **Products:** `prod_A` (Electronics), `prod_B` (Medical), `prod_C` (Raw Materials)
- **Max restock per order:** 200 units

---

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `timestep` | int | Current step (0–50) |
| `warehouse_stocks` | dict | Inventory per warehouse per product |
| `pending_orders` | list | Customer orders to fulfill this step |
| `in_transit` | list | Incoming restock shipments |
| `prices` | dict | Current market prices |
| `demand_forecast` | dict | Predicted demand per warehouse |

---

## 🏆 Tasks & Graders

| Difficulty | Task | Metric | Passing Score |
|-----------|------|--------|--------------|
| Easy | Minimum Stock Maintenance | % of (warehouse, product) pairs above 20 units | 0.70 |
| Medium | Order Fulfillment Rate | Fulfilled / Total orders, normalized at 70% target | 0.70 |
| Hard | Profit Optimization | Normalized total profit over episode | 0.50 |

All scores are in range **[0.0, 1.0]**.

---

## 💰 Reward Function

```
reward = clip((fulfillment - 0.1×holding_cost - 0.2×stockout_penalty - 0.01×restock_cost) / 240.0, 0.0, 1.0)
```

Reward is always in **[0.0, 1.0]** with partial credit for partial fulfillment.

---

## 🚀 Setup & Running

### Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the environment directly
python environment.py

# 3. Set required env variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="your_hf_token_here"

# 4. Run the inference script
python inference.py

# 5. Start the API server
python app.py
```

### Docker Build & Run

```bash
# Build
docker build -t supply-chain-openenv .

# Run with env variables
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" \
  -e HF_TOKEN="your_hf_token" \
  supply-chain-openenv
```

---

## 🌐 API Endpoints (HuggingFace Space)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check (returns 200) |
| POST | `/reset` | Reset environment, returns first observation |
| POST | `/step` | Execute action, returns obs/reward/done/info |
| GET | `/state` | Full internal state |
| GET | `/grade` | All grader scores |
| GET | `/openenv.yaml` | Spec file |

---

## 📊 Baseline Performance

Running the LLM-based baseline agent (`inference.py`) with default settings:

| Metric | Value |
|--------|-------|
| Average Reward | ~0.55–0.70 |
| Easy Score | ~0.75+ |
| Medium Score | ~0.65–0.75 |
| Hard Score | ~0.45–0.55 |

---

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | ✅ | API endpoint for LLM inference |
| `MODEL_NAME` | ✅ | Model identifier to use |
| `HF_TOKEN` | ✅ | HuggingFace API key |

---

## 📤 Inference Script Output Format

The `inference.py` script emits structured JSON logs to stdout:

```json
{"event": "START", "environment": "...", "model": "...", "max_steps": 50}
{"event": "STEP",  "step": 1, "action": {...}, "reward": 0.62, "done": false, "info": {...}}
...
{"event": "END",   "total_reward": 31.5, "grader_scores": {...}}
```

---

## 🏗️ Infrastructure

- Runtime limit: < 20 minutes  
- vCPU: 2 | Memory: 8GB  
- Python: 3.10  
- HuggingFace Spaces port: 7860
