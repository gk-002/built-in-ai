"""
OpenEnv: Supply Chain Inventory Management Environment
Real-world simulation of inventory management across a multi-warehouse supply chain.
An AI agent must balance stock levels, fulfill orders, and minimize costs.
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


# ─── Typed Models ────────────────────────────────────────────────────────────

@dataclass
class Product:
    id: str
    name: str
    holding_cost_per_unit: float   # cost per unit per timestep
    stockout_penalty: float        # penalty per unfulfilled unit
    reorder_lead_time: int         # timesteps to receive order

@dataclass
class Warehouse:
    id: str
    name: str
    capacity: int
    current_stock: Dict[str, int]  # product_id -> quantity

@dataclass
class Order:
    order_id: str
    product_id: str
    quantity: int
    timestep: int
    fulfilled: bool = False

@dataclass
class Action:
    restock_orders: Dict[str, Dict[str, int]]  # warehouse_id -> {product_id: qty}

@dataclass
class Observation:
    timestep: int
    warehouse_stocks: Dict[str, Dict[str, int]]   # warehouse_id -> {product_id: qty}
    pending_orders: List[Dict]                     # incoming customer orders
    in_transit: List[Dict]                         # pending restock shipments
    prices: Dict[str, float]                       # current product prices
    demand_forecast: Dict[str, Dict[str, float]]   # warehouse_id -> {product_id: expected_demand}

@dataclass
class StepResult:
    observation: Dict
    reward: float
    done: bool
    info: Dict


# ─── Environment ─────────────────────────────────────────────────────────────

class SupplyChainEnv:
    """
    OpenEnv-compliant Supply Chain Inventory Management Environment.

    Action Space:
        Dict of restock decisions per warehouse per product.
        e.g. {"wh_north": {"prod_A": 50, "prod_B": 20}}

    Observation Space:
        - warehouse_stocks: current inventory levels
        - pending_orders: customer orders to fulfill this step
        - in_transit: restock shipments arriving soon
        - prices: current unit prices
        - demand_forecast: predicted demand per warehouse per product

    Reward:
        Reward in [0.0, 1.0] based on:
        - Order fulfillment rate (higher = better)
        - Inventory holding cost (lower = better)
        - Stockout penalties (lower = better)
        - Overstock penalty (lower = better)
    """

    PRODUCTS = [
        Product("prod_A", "Electronic Components", 0.02, 0.5, 2),
        Product("prod_B", "Medical Supplies",      0.03, 0.8, 1),
        Product("prod_C", "Raw Materials",         0.01, 0.3, 3),
    ]

    WAREHOUSES = [
        Warehouse("wh_north", "North Hub",  500, {"prod_A": 100, "prod_B": 80, "prod_C": 150}),
        Warehouse("wh_south", "South Hub",  400, {"prod_A": 80,  "prod_B": 60, "prod_C": 120}),
        Warehouse("wh_east",  "East Hub",   300, {"prod_A": 60,  "prod_B": 40, "prod_C": 90}),
    ]

    MAX_TIMESTEPS = 50

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._timestep = 0
        self._done = False
        self._warehouses: Dict[str, Warehouse] = {}
        self._products: Dict[str, Product] = {}
        self._in_transit: List[Dict] = []       # {warehouse_id, product_id, qty, arrive_at}
        self._pending_orders: List[Order] = []
        self._cumulative_reward = 0.0
        self._episode_stats = {
            "total_orders": 0,
            "fulfilled_orders": 0,
            "total_holding_cost": 0.0,
            "total_stockout_penalty": 0.0,
        }

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Dict:
        """Reset environment to initial state. Returns initial observation."""
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)
        self._timestep = 0
        self._done = False
        self._in_transit = []
        self._cumulative_reward = 0.0
        self._episode_stats = {
            "total_orders": 0,
            "fulfilled_orders": 0,
            "total_holding_cost": 0.0,
            "total_stockout_penalty": 0.0,
        }

        # Deep-copy initial warehouse states
        self._warehouses = {
            wh.id: Warehouse(
                id=wh.id,
                name=wh.name,
                capacity=wh.capacity,
                current_stock=dict(wh.current_stock),
            )
            for wh in self.WAREHOUSES
        }
        self._products = {p.id: p for p in self.PRODUCTS}

        # Generate first batch of customer orders
        self._pending_orders = self._generate_orders()

        return self._get_observation()

    def step(self, action: Dict) -> Dict:
        """
        Execute one timestep.

        Args:
            action: {"restock_orders": {"wh_id": {"prod_id": qty, ...}, ...}}

        Returns:
            {"observation": ..., "reward": float, "done": bool, "info": {...}}
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # 1. Process restock action
        restock_cost = self._process_restock(action.get("restock_orders", {}))

        # 2. Receive in-transit shipments
        self._receive_shipments()

        # 3. Fulfill customer orders
        fulfillment_reward, stockout_penalty = self._fulfill_orders()

        # 4. Compute holding costs
        holding_cost = self._compute_holding_cost()

        # 5. Generate new orders for next step
        self._pending_orders = self._generate_orders()

        # 6. Advance timestep
        self._timestep += 1
        self._done = self._timestep >= self.MAX_TIMESTEPS

        # 7. Compute normalized reward [0.0, 1.0]
        reward = self._compute_reward(fulfillment_reward, holding_cost, stockout_penalty, restock_cost)
        self._cumulative_reward += reward

        self._episode_stats["total_holding_cost"] += holding_cost
        self._episode_stats["total_stockout_penalty"] += stockout_penalty

        info = {
            "timestep": self._timestep,
            "fulfillment_reward": fulfillment_reward,
            "holding_cost": holding_cost,
            "stockout_penalty": stockout_penalty,
            "restock_cost": restock_cost,
            "cumulative_reward": self._cumulative_reward,
            "episode_stats": self._episode_stats,
        }

        return {
            "observation": self._get_observation(),
            "reward": reward,
            "done": self._done,
            "info": info,
        }

    def state(self) -> Dict:
        """Return full internal state (for debugging/evaluation)."""
        return {
            "timestep": self._timestep,
            "done": self._done,
            "warehouses": {
                wh_id: {
                    "name": wh.name,
                    "capacity": wh.capacity,
                    "stock": wh.current_stock,
                    "utilization": sum(wh.current_stock.values()) / wh.capacity,
                }
                for wh_id, wh in self._warehouses.items()
            },
            "in_transit": self._in_transit,
            "pending_orders": [
                {"order_id": o.order_id, "product_id": o.product_id,
                 "quantity": o.quantity, "fulfilled": o.fulfilled}
                for o in self._pending_orders
            ],
            "cumulative_reward": self._cumulative_reward,
            "episode_stats": self._episode_stats,
        }

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _get_observation(self) -> Dict:
        obs = Observation(
            timestep=self._timestep,
            warehouse_stocks={
                wh_id: dict(wh.current_stock)
                for wh_id, wh in self._warehouses.items()
            },
            pending_orders=[
                {"order_id": o.order_id, "product_id": o.product_id,
                 "quantity": o.quantity, "warehouse_id": self._rng.choice(list(self._warehouses.keys()))}
                for o in self._pending_orders
            ],
            in_transit=list(self._in_transit),
            prices={p.id: round(self._rng.uniform(10, 50), 2) for p in self.PRODUCTS},
            demand_forecast=self._forecast_demand(),
        )
        return asdict(obs)

    def _generate_orders(self) -> List[Order]:
        orders = []
        n_orders = self._rng.randint(3, 8)
        for i in range(n_orders):
            product = self._rng.choice(self.PRODUCTS)
            qty = self._rng.randint(5, 30)
            orders.append(Order(
                order_id=f"ord_{self._timestep}_{i}",
                product_id=product.id,
                quantity=qty,
                timestep=self._timestep,
            ))
        self._episode_stats["total_orders"] += len(orders)
        return orders

    def _process_restock(self, restock_orders: Dict) -> float:
        """Schedule restock shipments. Returns total restock cost."""
        total_cost = 0.0
        for wh_id, products in restock_orders.items():
            if wh_id not in self._warehouses:
                continue
            for prod_id, qty in products.items():
                if prod_id not in self._products or qty <= 0:
                    continue
                qty = int(min(qty, 200))  # cap per order
                product = self._products[prod_id]
                arrive_at = self._timestep + product.reorder_lead_time
                self._in_transit.append({
                    "warehouse_id": wh_id,
                    "product_id": prod_id,
                    "quantity": qty,
                    "arrive_at": arrive_at,
                })
                total_cost += qty * 0.5  # flat restock cost
        return total_cost

    def _receive_shipments(self):
        remaining = []
        for shipment in self._in_transit:
            if shipment["arrive_at"] <= self._timestep:
                wh = self._warehouses[shipment["warehouse_id"]]
                prod_id = shipment["product_id"]
                qty = shipment["quantity"]
                current_total = sum(wh.current_stock.values())
                space = wh.capacity - current_total
                received = min(qty, space)
                wh.current_stock[prod_id] = wh.current_stock.get(prod_id, 0) + received
            else:
                remaining.append(shipment)
        self._in_transit = remaining

    def _fulfill_orders(self) -> Tuple[float, float]:
        fulfillment_reward = 0.0
        stockout_penalty = 0.0
        for order in self._pending_orders:
            product = self._products[order.product_id]
            # Try each warehouse
            fulfilled = False
            for wh in self._warehouses.values():
                available = wh.current_stock.get(order.product_id, 0)
                if available >= order.quantity:
                    wh.current_stock[order.product_id] -= order.quantity
                    order.fulfilled = True
                    fulfillment_reward += order.quantity * 1.0
                    fulfilled = True
                    self._episode_stats["fulfilled_orders"] += 1
                    break
                elif available > 0:
                    # Partial fulfillment
                    partial = available
                    wh.current_stock[order.product_id] = 0
                    fulfillment_reward += partial * 0.5
                    unfulfilled = order.quantity - partial
                    stockout_penalty += unfulfilled * product.stockout_penalty
                    fulfilled = True
                    break
            if not fulfilled:
                stockout_penalty += order.quantity * product.stockout_penalty
        return fulfillment_reward, stockout_penalty

    def _compute_holding_cost(self) -> float:
        cost = 0.0
        for wh in self._warehouses.values():
            for prod_id, qty in wh.current_stock.items():
                product = self._products[prod_id]
                cost += qty * product.holding_cost_per_unit
        return cost

    def _compute_reward(self, fulfillment: float, holding: float,
                        stockout: float, restock: float) -> float:
        """Normalized reward in [0.0, 1.0]."""
        gross = fulfillment - holding * 0.1 - stockout * 0.2 - restock * 0.01
        # Normalize: max possible fulfillment per step ~240 units
        normalized = gross / 240.0
        return float(max(0.0, min(1.0, normalized)))

    def _forecast_demand(self) -> Dict:
        forecast = {}
        for wh_id in self._warehouses:
            forecast[wh_id] = {}
            for prod in self.PRODUCTS:
                # Simple noisy forecast
                base = self._rng.uniform(10, 40)
                noise = self._rng.gauss(0, 5)
                forecast[wh_id][prod.id] = round(max(0, base + noise), 1)
        return forecast


# ─── Task Graders ─────────────────────────────────────────────────────────────

class EasyGrader:
    """
    Easy Task: Maintain minimum stock levels across all warehouses.
    Score = fraction of (warehouse, product) pairs above minimum threshold.
    """
    MINIMUM_STOCK = 20

    def grade(self, env_state: Dict) -> float:
        warehouses = env_state.get("warehouses", {})
        total_pairs = 0
        above_min = 0
        for wh_data in warehouses.values():
            for prod_id, qty in wh_data["stock"].items():
                total_pairs += 1
                if qty >= self.MINIMUM_STOCK:
                    above_min += 1
        if total_pairs == 0:
            return 0.0
        return round(above_min / total_pairs, 4)


class MediumGrader:
    """
    Medium Task: Achieve >70% order fulfillment rate over an episode.
    Score = min(fulfilled_orders / total_orders, 1.0)
    """
    TARGET_RATE = 0.70

    def grade(self, env_state: Dict) -> float:
        stats = env_state.get("episode_stats", {})
        total = stats.get("total_orders", 0)
        fulfilled = stats.get("fulfilled_orders", 0)
        if total == 0:
            return 0.0
        rate = fulfilled / total
        # Scale: 0 at 0%, 1.0 at TARGET_RATE+
        score = min(rate / self.TARGET_RATE, 1.0)
        return round(score, 4)


class HardGrader:
    """
    Hard Task: Optimize total profit (fulfillment revenue - all costs).
    Score = normalized profit over episode, clipped to [0.0, 1.0].
    """
    MAX_EXPECTED_PROFIT = 5000.0

    def grade(self, env_state: Dict) -> float:
        stats = env_state.get("episode_stats", {})
        fulfilled = stats.get("fulfilled_orders", 0)
        holding = stats.get("total_holding_cost", 0.0)
        stockout = stats.get("total_stockout_penalty", 0.0)
        # Rough profit estimate
        revenue = fulfilled * 20.0       # avg revenue per fulfilled order
        costs = holding + stockout
        profit = revenue - costs
        score = profit / self.MAX_EXPECTED_PROFIT
        return round(max(0.0, min(1.0, score)), 4)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = SupplyChainEnv(seed=42)
    obs = env.reset()
    print("=== Reset ===")
    print(json.dumps(obs, indent=2))

    # Random agent
    for step in range(5):
        action = {
            "restock_orders": {
                "wh_north": {"prod_A": 30, "prod_B": 20},
                "wh_south": {"prod_B": 15, "prod_C": 25},
            }
        }
        result = env.step(action)
        print(f"\nStep {step+1}: reward={result['reward']:.4f}, done={result['done']}")

    state = env.state()
    print("\n=== State ===")
    print(json.dumps(state, indent=2))

    # Graders
    easy  = EasyGrader().grade(state)
    medium = MediumGrader().grade(state)
    hard  = HardGrader().grade(state)
    print(f"\nGrader Scores — Easy: {easy}, Medium: {medium}, Hard: {hard}")
