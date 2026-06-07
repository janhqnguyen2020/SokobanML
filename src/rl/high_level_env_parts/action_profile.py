"""
Action helpers for the high-level Sokoban environment.
"""

from src.planners.deadlock import has_deadlock
from src.planners.heuristics import hungarian_matching
from src.rl.high_level_env_parts.movement import (
    blocked_push,
    get_reachable_parents,
    new_box_positions,
    push_positions,
    reconstruct_path,
)
from src.rl.high_level_env_parts.state import (
    count_boxes_on_target,
    feasible_goals_per_box,
    read_board,
)

TOP_K_ACTIONS = 8
FLEXIBILITY_SCORE_WEIGHT = 20.0
FORCED_BOX_SCORE_PENALTY = 15.0
FUTURE_PUSH_SCORE_WEIGHT = 2.0
SOLVED_BOX_SCORE_WEIGHT = 100.0
MATCHING_COST_SCORE_WEIGHT = 1.0
GOAL_EXIT_SCORE_PENALTY = 12.0


def build_macro_action(
    reachable_parents,
    selected_box_position,
    box_index,
    direction_index,
    box_positions,
    wall_positions,
    direction_deltas,
):
    """
    Build one macro action when the requested push is physically possible.
    """
    push_direction = direction_index + 1
    player_push_position, box_destination_position = push_positions(
        direction_deltas,
        selected_box_position,
        push_direction,
    )
    if blocked_push(
        player_push_position,
        box_destination_position,
        box_positions,
        wall_positions,
        reachable_parents,
    ):
        return None
    return macro_action_data(
        selected_box_position,
        box_index,
        direction_index,
        push_direction,
        reachable_parents,
        player_push_position,
        box_positions,
        box_destination_position,
    )


def macro_action_data(
    selected_box_position,
    box_index,
    direction_index,
    push_direction,
    reachable_parents,
    player_push_position,
    box_positions,
    box_destination_position,
):
    """
    Store one legal macro push in a small beginner-friendly dictionary.
    """
    return {
        "macro_action": box_index * 4 + direction_index,
        "box_index": box_index,
        "box_position": selected_box_position,
        "direction": push_direction,
        "walk_actions": reconstruct_path(reachable_parents, player_push_position),
        "new_player_pos": selected_box_position,
        "new_boxes": new_box_positions(
            box_positions,
            selected_box_position,
            box_destination_position,
        ),
    }


def add_box_pushes(
    valid_action_data,
    reachable_parents,
    selected_box_position,
    box_index,
    box_positions,
    wall_positions,
    direction_deltas,
):
    """
    Add every physically legal push for one chosen box.
    """
    for direction_index in range(4):
        action_data = build_macro_action(
            reachable_parents,
            selected_box_position,
            box_index,
            direction_index,
            box_positions,
            wall_positions,
            direction_deltas,
        )
        if action_data is not None:
            valid_action_data[action_data["macro_action"]] = action_data


def build_physical_action_data(
    player_pos,
    box_positions,
    wall_positions,
    direction_deltas,
    goal_positions=None,
):
    """
    Return every one-push action the player can physically execute.
    """
    reachable_parents = get_reachable_parents(
        player_pos,
        box_positions,
        wall_positions,
        direction_deltas,
    )
    valid_action_data = {}
    for box_index, selected_box_position in enumerate(sorted(list(box_positions))):
        add_box_pushes(
            valid_action_data,
            reachable_parents,
            selected_box_position,
            box_index,
            box_positions,
            wall_positions,
            direction_deltas,
        )
    return valid_action_data


def future_push_count(
    action_data,
    wall_positions,
    direction_deltas,
    goal_positions=None,
):
    """
    Count how many physical pushes remain after one candidate move.
    """
    next_actions = build_physical_action_data(
        action_data["new_player_pos"],
        action_data["new_boxes"],
        wall_positions,
        direction_deltas,
        goal_positions,
    )
    return len(next_actions)


def safe_action_data_entry(
    action_data,
    goal_positions,
    wall_positions,
    dead_squares,
    num_boxes,
    direction_deltas,
):
    """
    Return enriched action data unless the move immediately deadlocks.
    """
    if has_deadlock(
        action_data["new_boxes"],
        dead_squares,
        wall_positions,
        goal_positions,
    ):
        return None
    enriched_action_data = dict(action_data)
    solved = count_boxes_on_target(action_data["new_boxes"], goal_positions) == num_boxes
    enriched_action_data["future_physical_pushes"] = 0 if solved else future_push_count(
        action_data,
        wall_positions,
        direction_deltas,
        goal_positions,
    )
    return enriched_action_data


def is_viable_action(action_data, goal_positions, num_boxes):
    """
    Return True when the move solves the puzzle or keeps future pushes open.
    """
    solved = count_boxes_on_target(action_data["new_boxes"], goal_positions) == num_boxes
    return solved or action_data["future_physical_pushes"] > 0


def split_safe_actions(
    physical_action_data,
    goal_positions,
    wall_positions,
    dead_squares,
    num_boxes,
    direction_deltas,
):
    """
    Split physical pushes into safe pushes and more promising viable pushes.
    """
    safe_action_data = {}
    viable_safe_action_data = {}
    for action, action_data in physical_action_data.items():
        enriched_action_data = safe_action_data_entry(
            action_data,
            goal_positions,
            wall_positions,
            dead_squares,
            num_boxes,
            direction_deltas,
        )
        if enriched_action_data is None:
            continue
        safe_action_data[action] = enriched_action_data
        if is_viable_action(enriched_action_data, goal_positions, num_boxes):
            viable_safe_action_data[action] = enriched_action_data
    return safe_action_data, viable_safe_action_data


def action_profile_summary(
    selected,
    pool_name,
    physical_action_data,
    safe_action_data,
    viable_safe_action_data,
):
    """
    Return one consistent action-profile dictionary for the observation code.
    """
    return {
        "selected": selected,
        "pool_name": pool_name,
        "physical_count": len(physical_action_data),
        "safe_count": len(safe_action_data),
        "viable_count": len(viable_safe_action_data),
    }


def build_action_profile(
    player_pos,
    box_positions,
    goal_positions,
    wall_positions,
    dead_squares,
    num_boxes,
    direction,
):
    """
    Build the selected macro-action pool for the current board.
    """
    physical_action_data = build_physical_action_data(
        player_pos,
        box_positions,
        wall_positions,
        direction,
        goal_positions,
    )
    safe_action_data, viable_safe_action_data = split_safe_actions(
        physical_action_data,
        goal_positions,
        wall_positions,
        dead_squares,
        num_boxes,
        direction,
    )
    selected_actions = select_scored_viable_actions(
        viable_safe_action_data,
        box_positions,
        goal_positions,
        wall_positions,
        direction,
    )
    if selected_actions:
        return action_profile_summary(
            selected_actions,
            "viable_safe_scored",
            physical_action_data,
            safe_action_data,
            viable_safe_action_data,
        )
    if safe_action_data:
        return action_profile_summary(
            safe_action_data,
            "safe",
            physical_action_data,
            safe_action_data,
            viable_safe_action_data,
        )
    return action_profile_summary(
        {},
        "none",
        physical_action_data,
        safe_action_data,
        viable_safe_action_data,
    )


def select_scored_viable_actions(
    viable_safe_action_data,
    box_positions,
    goal_positions,
    wall_positions,
    direction,
):
    """
    Keep only viable actions that preserve a consistent box-to-goal future.
    """
    if not viable_safe_action_data:
        return {}
    domains_before = goal_domains_for_boxes(
        box_positions,
        goal_positions,
        wall_positions,
        box_positions,
        direction,
    )
    filtered_actions = matching_safe_actions(
        viable_safe_action_data,
        goal_positions,
        wall_positions,
        direction,
    )
    if not filtered_actions:
        return {}
    return top_scored_actions(
        filtered_actions,
        domains_before,
        goal_positions,
    )


def goal_domains_for_boxes(
    boxes_to_score,
    goal_positions,
    wall_positions,
    all_box_positions,
    direction,
):
    """
    Return the feasible goal list for every box in one board state.
    """
    return {
        box_position: feasible_goals_per_box(
            box_position,
            goal_positions,
            wall_positions,
            all_box_positions,
            direction,
        )
        for box_position in boxes_to_score
    }


def matching_safe_actions(
    viable_safe_action_data,
    goal_positions,
    wall_positions,
    direction,
):
    """
    Keep only actions whose box-goal domains still admit a full assignment.
    """
    filtered_actions = {}
    for action, action_data in viable_safe_action_data.items():
        if destroys_box_domain(action_data, goal_positions, wall_positions, direction):
            continue
        filtered_actions[action] = action_with_goal_domains(
            action_data,
            action_goal_domains(action_data, goal_positions, wall_positions, direction),
        )
    return filtered_actions


def action_with_goal_domains(action_data, goal_domains):
    """
    Attach the computed goal domains so later scoring can reuse them cheaply.
    """
    enriched_action_data = dict(action_data)
    enriched_action_data["goal_domains"] = goal_domains
    return enriched_action_data


def action_goal_domains(action_data, goal_positions, wall_positions, direction):
    """
    Return the feasible goal domains after one candidate push.
    """
    return goal_domains_for_boxes(
        action_data["new_boxes"],
        goal_positions,
        wall_positions,
        action_data["new_boxes"],
        direction,
    )


def destroys_box_domain(action_data, goal_positions, wall_positions, direction):
    """
    Return True when a move makes some box-goal assignment impossible.
    """
    goal_domains = action_goal_domains(
        action_data,
        goal_positions,
        wall_positions,
        direction,
    )
    return missing_goal_domain(goal_domains) or not has_complete_matching(goal_domains)


def missing_goal_domain(goal_domains):
    """
    Return True when some box has no feasible goal left at all.
    """
    return any(len(domain) == 0 for domain in goal_domains.values())


def has_complete_matching(goal_domains):
    """
    Return True when every box can still be assigned to a distinct goal.
    """
    sorted_boxes = sorted(goal_domains.keys())
    return match_goals_for_boxes(sorted_boxes, goal_domains, set(), 0)


def match_goals_for_boxes(sorted_boxes, goal_domains, used_goals, box_index):
    """
    Try to assign one distinct goal per box with a small depth-first search.
    """
    if box_index == len(sorted_boxes):
        return True
    current_box = sorted_boxes[box_index]
    for goal_position in goal_domains[current_box]:
        if goal_position in used_goals:
            continue
        used_goals.add(goal_position)
        if match_goals_for_boxes(sorted_boxes, goal_domains, used_goals, box_index + 1):
            return True
        used_goals.remove(goal_position)
    return False


def top_scored_actions(filtered_actions, domains_before, goal_positions):
    """
    Keep only the highest-scoring subset of still-promising macro actions.
    """
    sorted_actions = sorted(
        filtered_actions.items(),
        key=lambda item: action_score(item[1], domains_before, goal_positions),
        reverse=True,
    )
    return dict(sorted_actions[:TOP_K_ACTIONS])


def action_score(action_data, domains_before, goal_positions):
    """
    Score one action by balancing flexibility, progress, and assignment quality.
    """
    domains_after = action_data["goal_domains"]
    flexibility_gain = total_domain_size(domains_after) - total_domain_size(domains_before)
    forced_boxes = forced_box_count(domains_after)
    solved_boxes = count_boxes_on_target(action_data["new_boxes"], goal_positions)
    future_pushes = action_data["future_physical_pushes"]
    matching_cost = hungarian_matching(action_data["new_boxes"], goal_positions)
    moved_goal_box = moved_box_off_goal(action_data, goal_positions)
    return (
        FLEXIBILITY_SCORE_WEIGHT * flexibility_gain
        - FORCED_BOX_SCORE_PENALTY * forced_boxes
        + FUTURE_PUSH_SCORE_WEIGHT * future_pushes
        + SOLVED_BOX_SCORE_WEIGHT * solved_boxes
        - MATCHING_COST_SCORE_WEIGHT * matching_cost
        - GOAL_EXIT_SCORE_PENALTY * moved_goal_box
    )


def total_domain_size(goal_domains):
    """
    Return how many total box-goal options remain in one board state.
    """
    return sum(len(domain) for domain in goal_domains.values())


def forced_box_count(goal_domains):
    """
    Count how many boxes are already forced to exactly one remaining goal.
    """
    return sum(1 for domain in goal_domains.values() if len(domain) == 1)


def moved_box_off_goal(action_data, goal_positions):
    """
    Return one when the chosen box started on a goal and zero otherwise.
    """
    if action_data["box_position"] in goal_positions:
        return 1
    return 0


def board_profile(env, dead_squares, num_boxes, direction):
    """
    Read the board and build the current action profile.
    """
    player_pos, box_positions, goal_positions, wall_positions = read_board(env)
    action_profile = build_action_profile(
        player_pos,
        box_positions,
        goal_positions,
        wall_positions,
        dead_squares,
        num_boxes,
        direction,
    )
    return player_pos, box_positions, goal_positions, wall_positions, action_profile
