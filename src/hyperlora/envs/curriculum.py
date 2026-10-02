from __future__ import annotations

STAGE_KWARGS = {
    1: {"ai_red_agents": True, "disable_ai_red": True, "dense_reward": True,
        "pos_shaping_factor_ball_goal": 100.0, "pos_shaping_factor_agent_ball": 10.0},
    2: {"ai_red_agents": True, "disable_ai_red": False, "dense_reward": True,
        "ai_strength": (1.0, 0.5), "ai_decision_strength": (1.0, 0.5),
        "ai_precision_strength": (1.0, 0.5),
        "pos_shaping_factor_ball_goal": 50.0, "pos_shaping_factor_agent_ball": 5.0},
    3: {"ai_red_agents": True, "disable_ai_red": False, "dense_reward": True,
        "ai_strength": (1.0, 1.0), "ai_decision_strength": (1.0, 1.0),
        "ai_precision_strength": (1.0, 1.0),
        "pos_shaping_factor_ball_goal": 25.0, "pos_shaping_factor_agent_ball": 2.5},
}


def football_stage(episode: int, stage1_episodes: int, stage2_episodes: int) -> int:
    if episode < stage1_episodes:
        return 1
    if episode < stage1_episodes + stage2_episodes:
        return 2
    return 3

