"""
app.py — FastAPI server for HuggingFace Spaces deployment.
Exposes /reset, /step, /state endpoints as required by OpenEnv spec.
Automated ping must return 200 and respond to reset().
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
from environment import SupplyChainEnv, EasyGrader, MediumGrader, HardGrader

app = FastAPI(
    title="Supply Chain Inventory Management - OpenEnv",
    description="Real-world supply chain environment for AI agent training.",
    version="1.0.0",
)

# Global environment instance (single session for evaluation)
_env: Optional[SupplyChainEnv] = None


def get_env() -> SupplyChainEnv:
    global _env
    if _env is None:
        _env = SupplyChainEnv(seed=42)
    return _env


# ─── Request Models ───────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action: Dict[str, Any]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — must return 200."""
    return {
        "status": "ok",
        "environment": "supply-chain-inventory-management",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grade"],
    }


@app.get("/health")
def health():
    """Explicit health endpoint."""
    return {"status": "healthy"}


@app.post("/reset")
def reset(seed: Optional[int] = None):
    """
    Reset the environment to its initial state.
    Returns the first observation.
    """
    global _env
    _env = SupplyChainEnv(seed=seed if seed is not None else 42)
    obs = _env.reset()
    return JSONResponse(content={"observation": obs, "status": "reset"})


@app.post("/step")
def step(request: StepRequest):
    """
    Execute one environment step.
    Body: {"action": {"restock_orders": {...}}}
    Returns: {observation, reward, done, info}
    """
    env = get_env()
    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode."
        )
    try:
        result = env.step(request.action)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """
    Return full internal environment state.
    Used for debugging and evaluation grading.
    """
    env = get_env()
    return JSONResponse(content=env.state())


@app.get("/grade")
def grade():
    """
    Run all three graders on current episode state.
    Returns scores for easy, medium, and hard tasks.
    """
    env = get_env()
    current_state = env.state()

    easy_score   = EasyGrader().grade(current_state)
    medium_score = MediumGrader().grade(current_state)
    hard_score   = HardGrader().grade(current_state)

    return JSONResponse(content={
        "grader_scores": {
            "easy":   {"score": easy_score,   "range": [0.0, 1.0], "passing": 0.7},
            "medium": {"score": medium_score, "range": [0.0, 1.0], "passing": 0.7},
            "hard":   {"score": hard_score,   "range": [0.0, 1.0], "passing": 0.5},
        },
        "episode_stats": current_state.get("episode_stats", {}),
        "timestep": current_state.get("timestep", 0),
    })


@app.get("/openenv.yaml")
def get_yaml():
    """Serve the openenv.yaml spec file."""
    try:
        with open("openenv.yaml", "r") as f:
            content = f.read()
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=content, media_type="text/yaml")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,          # HuggingFace Spaces default port
        reload=False,
        log_level="info",
    )
