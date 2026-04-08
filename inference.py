"""
inference.py — Baseline Inference Script for Supply Chain OpenEnv
Structured stdout logging: [START], [STEP], [END] format as required.
Uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN env variables.
"""

import os
import sys
import json
import time
from openai import OpenAI
from environment import SupplyChainEnv, EasyGrader, MediumGrader, HardGrader

# ─── Environment Config ───────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# ─── OpenAI Client (pointing to HuggingFace) ─────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an expert supply chain manager AI agent.
You control inventory restocking decisions across three warehouses:
- wh_north (capacity: 500)
- wh_south (capacity: 400)
- wh_east  (capacity: 300)

Products:
- prod_A: Electronic Components (lead time 2 steps, stockout penalty high)
- prod_B: Medical Supplies      (lead time 1 step,  stockout penalty very high)
- prod_C: Raw Materials         (lead time 3 steps, stockout penalty moderate)

Your goal: Fulfill as many customer orders as possible while minimizing holding costs.

RULES:
- Max 200 units per restock order per product per warehouse
- You must respond ONLY with valid JSON, no explanation
- JSON format:
{
  "restock_orders": {
    "wh_north": {"prod_A": <int>, "prod_B": <int>, "prod_C": <int>},
    "wh_south": {"prod_A": <int>, "prod_B": <int>, "prod_C": <int>},
    "wh_east":  {"prod_A": <int>, "prod_B": <int>, "prod_C": <int>}
  }
}
- Set quantity to 0 if no restock needed
- Order proactively given lead times (prod_C has 3-step lead time!)
"""

def get_llm_action(observation: dict, step_num: int) -> dict:
    """Query the LLM for an action given the current observation."""
    obs_summary = {
        "timestep": observation["timestep"],
        "stocks": observation["warehouse_stocks"],
        "pending_orders": observation["pending_orders"],
        "in_transit": observation["in_transit"],
        "demand_forecast": observation["demand_forecast"],
    }

    user_message = f"""
Current environment state (step {step_num}):
{json.dumps(obs_summary, indent=2)}

Analyze the stock levels, pending orders, and demand forecast.
Decide how many units of each product to restock in each warehouse.
Respond ONLY with the JSON action. No explanation.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action = json.loads(raw)
        return action

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: safe default action
        return {
            "restock_orders": {
                "wh_north": {"prod_A": 20, "prod_B": 15, "prod_C": 25},
                "wh_south": {"prod_A": 15, "prod_B": 10, "prod_C": 20},
                "wh_east":  {"prod_A": 10, "prod_B": 8,  "prod_C": 15},
            }
        }


def run_inference():
    """Main inference loop with required structured stdout logging."""
    env = SupplyChainEnv(seed=42)
    easy_grader   = EasyGrader()
    medium_grader = MediumGrader()
    hard_grader   = HardGrader()

    # ── [START] ───────────────────────────────────────────────────────────────
    print(json.dumps({
        "event": "START",
        "environment": "supply-chain-inventory-management",
        "model": MODEL_NAME,
        "max_steps": SupplyChainEnv.MAX_TIMESTEPS,
        "timestamp": time.time(),
    }))
    sys.stdout.flush()

    obs = env.reset()
    total_reward = 0.0
    steps_data = []

    for step_num in range(SupplyChainEnv.MAX_TIMESTEPS):
        # Get LLM action
        action = get_llm_action(obs, step_num)

        # Execute step
        result = env.step(action)
        obs      = result["observation"]
        reward   = result["reward"]
        done     = result["done"]
        info     = result["info"]
        total_reward += reward

        step_record = {
            "event":   "STEP",
            "step":    step_num + 1,
            "action":  action,
            "reward":  round(reward, 6),
            "done":    done,
            "info": {
                "fulfillment_reward":  round(info["fulfillment_reward"], 4),
                "holding_cost":        round(info["holding_cost"], 4),
                "stockout_penalty":    round(info["stockout_penalty"], 4),
                "restock_cost":        round(info["restock_cost"], 4),
                "cumulative_reward":   round(info["cumulative_reward"], 6),
            },
        }

        # ── [STEP] ─────────────────────────────────────────────────────────
        print(json.dumps(step_record))
        sys.stdout.flush()

        steps_data.append(step_record)

        if done:
            break

    # Final grader scores
    final_state = env.state()
    easy_score   = easy_grader.grade(final_state)
    medium_score = medium_grader.grade(final_state)
    hard_score   = hard_grader.grade(final_state)

    stats = final_state["episode_stats"]
    fulfillment_rate = (
        stats["fulfilled_orders"] / stats["total_orders"]
        if stats["total_orders"] > 0 else 0.0
    )

    # ── [END] ────────────────────────────────────────────────────────────────
    print(json.dumps({
        "event": "END",
        "total_steps":       step_num + 1,
        "total_reward":      round(total_reward, 6),
        "average_reward":    round(total_reward / (step_num + 1), 6),
        "fulfillment_rate":  round(fulfillment_rate, 4),
        "grader_scores": {
            "easy":   easy_score,
            "medium": medium_score,
            "hard":   hard_score,
        },
        "episode_stats": {
            "total_orders":          stats["total_orders"],
            "fulfilled_orders":      stats["fulfilled_orders"],
            "total_holding_cost":    round(stats["total_holding_cost"], 4),
            "total_stockout_penalty": round(stats["total_stockout_penalty"], 4),
        },
        "timestamp": time.time(),
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    run_inference()
