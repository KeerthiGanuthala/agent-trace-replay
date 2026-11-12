"""Analytic trajectory survival model for Experiment 5 (stretch goal).

Computes the exact probability that a fresh live rerun after `delta_epochs` of
world drift reproduces a recorded trajectory, decomposing survival into
independent keyed factors:
1. Stochastic policy choices & tie-break draws.
2. Transient flakiness pattern across tool attempts.
3. Deduplicated item price stability (when budget or price checks are on path).
4. Stock availability bookkeeping after prior reservations.
5. Service-time deadline survival via dynamic programming over step latency distributions.
"""

from __future__ import annotations

from agent_trace_replay.agent import AgentConfig, RunOutcome
from agent_trace_replay.world import OrderDeskWorld


def _single_epoch_stay_prob(drift_rate: float) -> float:
    return 1.0 - drift_rate


def price_trajectory_match_prob(
    world_at_record: OrderDeskWorld,
    outcome: RunOutcome,
    delta_epochs: int,
) -> float:
    """Probability that price drift over `delta_epochs` keeps the trajectory in the same budget regime."""
    if delta_epochs == 0:
        # Even at delta_epochs=0, a fresh attempt tag draws fresh RNG for flakiness/choices/clock,
        # which is handled by the other factors; price itself is identical at epoch 0.
        return 1.0

    # Check if price step was reached
    price_steps = [s for s in outcome.steps if s.node == "price"]
    if not price_steps:
        return 1.0

    # If the price step failed due to retry exhaustion before computing quote_total, price value didn't branch
    p_step = price_steps[0]
    if p_step.next_node == "give_up":
        return 1.0

    # Deduplicated per item: even if price is read twice (e.g. recheck_price_first),
    # both reads observe the same item at the same epoch E.
    r = world_at_record.drift_rate
    step = world_at_record.drift_step
    p_up = r * 0.58
    p_down = r * 0.42
    p_stay = _single_epoch_stay_prob(r)

    # DP over price delta after `delta_epochs` steps
    dist: dict[int, float] = {0: 1.0}
    for _ in range(delta_epochs):
        nxt: dict[int, float] = {}
        for delta, prob in dist.items():
            nxt[delta] = nxt.get(delta, 0.0) + prob * p_stay
            nxt[delta + step] = nxt.get(delta + step, 0.0) + prob * p_up
            nxt[delta - step] = nxt.get(delta - step, 0.0) + prob * p_down
        dist = nxt

    # For exact state fingerprint reproduction at L3, unit_price in final_state must match recorded unit_price!
    # Therefore delta must be 0 at the debug epoch.
    return dist.get(0, 0.0)


def attempt_pattern_match_prob(outcome: RunOutcome, flaky_rate: float) -> float:
    """Probability that a fresh attempt reproduces the exact transient error/success pattern."""
    prob = 1.0
    for s in outcome.steps:
        for att in s.tool_attempts:
            if att.effect == "PURE":
                continue
            if att.error is not None:
                prob *= flaky_rate
            else:
                prob *= 1.0 - flaky_rate
    return prob


def choice_and_tiebreak_match_prob(outcome: RunOutcome) -> float:
    """Probability that ambient tie-break and stochastic choice points match the recorded run."""
    prob = 1.0
    for s in outcome.steps:
        if s.tie_break_index is not None:
            prob *= 1.0 / 3.0  # 3 related docs
        if s.choice_made is not None:
            if s.node == "stock":
                # Only stochastic when stock was tight (available_stock <= qty + 2)
                avail = outcome.final_state.get("available_stock")
                qty = outcome.final_state.get("qty", 0)
                if avail is not None and int(avail) <= int(qty) + 2:
                    prob *= 0.5
            elif s.node == "notify":
                prob *= 0.5
    return prob


def stock_bookkeeping_match_prob(
    initial_stock: int, qty: int, prior_reserved_qty: int, outcome: RunOutcome
) -> float:
    """Account for stock mutation if the first run already reserved stock."""
    stock_steps = [s for s in outcome.steps if s.node == "stock"]
    if not stock_steps or stock_steps[0].next_node == "give_up":
        return 1.0
    # If the first run mutated stock via reserve_stock, available_stock on a rerun against the mutated
    # world will differ, immediately changing the state fingerprint at the `stock` node.
    if prior_reserved_qty > 0:
        return 0.0
    return 1.0


def deadline_guard_survival_prob(outcome: RunOutcome, config: AgentConfig) -> float:
    """Dynamic program over discrete latency buckets (5ms grid) for deadline guard agreement."""
    grid_ms = 5.0
    max_bucket = int(config.deadline_ms // grid_ms) + 40
    # dp[b] = probability mass at bucket b
    dp: dict[int, float] = {0: 1.0}

    p_hiccup = config.hiccup_prob
    for s in outcome.steps:
        if s.node in ("finish", "give_up"):
            break
        # Check if recorded run timed out at the start of this step
        recorded_timed_out = s.next_node == "give_up" and outcome.failure_kind == "deadline_exceeded"
        if recorded_timed_out:
            # Probability we are currently above deadline_ms
            return sum(prob for b, prob in dp.items() if b * grid_ms > config.deadline_ms)

        # Condition on surviving the deadline guard at start of step
        surviving = {b: prob for b, prob in dp.items() if b * grid_ms <= config.deadline_ms}
        total_surv = sum(surviving.values())
        if total_surv <= 0.0:
            return 0.0

        # Convolve with each tool attempt's latency distribution in this step
        nxt_dp = dict(surviving)
        if not s.tool_attempts and s.node == "plan":
            nxt_dp = {b + 1: p for b, p in nxt_dp.items()}
        for att in s.tool_attempts:
            base_b = max(1, int(round((att.service_ms if att.service_ms < 45 else 20.0) / grid_ms)))
            backoff_b = int(round(att.backoff_ms / grid_ms))
            hiccup_b = int(round(config.hiccup_ms / grid_ms))
            updated: dict[int, float] = {}
            for b, prob in nxt_dp.items():
                # No hiccup + average jitter (1 bucket)
                b_norm = min(max_bucket, b + base_b + 1 + backoff_b)
                updated[b_norm] = updated.get(b_norm, 0.0) + prob * (1.0 - p_hiccup)
                # Hiccup
                b_hic = min(max_bucket, b + base_b + 1 + hiccup_b + backoff_b)
                updated[b_hic] = updated.get(b_hic, 0.0) + prob * p_hiccup
            nxt_dp = updated
        dp = nxt_dp

    return sum(dp.values())


def predict_cohort_reproduction_rate(
    worlds_and_outcomes: list[tuple[OrderDeskWorld, RunOutcome, int]],
    delta_epochs: int,
    config: AgentConfig,
) -> dict[str, float]:
    """Compare the structured keyed survival model against a naive per-step marginal hazard model."""
    if not worlds_and_outcomes:
        return {"analytic_branch_rate": 0.0, "naive_hazard_rate": 0.0}

    analytic_probs: list[float] = []
    naive_probs: list[float] = []

    for world0, outcome, prior_reserved in worlds_and_outcomes:
        p_att = attempt_pattern_match_prob(outcome, world0.flaky_rate)
        p_choice = choice_and_tiebreak_match_prob(outcome)
        p_price = price_trajectory_match_prob(world0, outcome, delta_epochs)
        p_stock = stock_bookkeeping_match_prob(
            world0.stock[world0.orders[world0.primary_order_id].item],
            world0.orders[world0.primary_order_id].qty,
            prior_reserved,
            outcome,
        )
        p_dead = deadline_guard_survival_prob(outcome, config)
        analytic_probs.append(p_att * p_choice * p_price * p_stock * p_dead)

        # Naive model: multiplies marginal per-step hazard (assuming every step independently
        # suffers drift + flakiness + choice hazard without deduplicating item price or effect state)
        per_step_survival = (1.0 - world0.flaky_rate) * (1.0 - world0.drift_rate * 0.5) * 0.78
        naive_probs.append(per_step_survival ** (len(outcome.steps) + delta_epochs))

    return {
        "analytic_trajectory_rate": round(sum(analytic_probs) / len(analytic_probs), 4),
        "naive_hazard_rate": round(sum(naive_probs) / len(naive_probs), 4),
    }
