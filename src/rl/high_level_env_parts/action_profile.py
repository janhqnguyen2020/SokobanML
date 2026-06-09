# src/rl/high_level_env_parts/action_profile.py
"""
Action helpers for the high-level Sokoban environment
"""

from src.planners.deadlock import has_deadlock
from src.rl.high_level_env_parts.state import count_boxes_on_target, read_board
from src.rl.high_level_env_parts.movement import (
    blocked_push,
    get_reachable_parents,
    new_box_positions,
    push_positions,
    reconstruct_path,
)
from src.rl.high_level_env_parts.state import feasible_goals_per_box

def build_macro_action(reachable_parents, selected_box_position, box_index, direction_index, box_positions, wall_positions, direction_deltas):
    """
    Build one macro action when the push is physically legal
    """
    push_direction = direction_index + 1
    player_push_position, box_destination_position = push_positions(direction_deltas, selected_box_position, push_direction)
    if blocked_push(player_push_position, box_destination_position, box_positions, wall_positions, reachable_parents):
        return None
    return {
        "macro_action": box_index * 4 + direction_index,
        # for previous move tracking to avoid RL oscillations
        "box_index": box_index,
        "box_position": selected_box_position,
        "direction": push_direction,
        "walk_actions": reconstruct_path(reachable_parents, player_push_position),
        "new_player_pos": selected_box_position,
        "new_boxes": new_box_positions(box_positions, selected_box_position, box_destination_position),
    }


def add_box_pushes(valid_action_data, reachable_parents, selected_box_position, box_index, box_positions, wall_positions, direction_deltas):
    """
    Add every legal push for one chosen box
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


def build_physical_action_data(player_pos, box_positions, wall_positions, direction_deltas, goal_positions=None):
    """
    Return every one-push action the player can physically execute.
    Boxes already on a goal are skipped — once placed they are frozen.
    """
    reachable_parents = get_reachable_parents(player_pos, box_positions, wall_positions, direction_deltas)
    valid_action_data = {}
    for box_index, selected_box_position in enumerate(sorted(list(box_positions))):
        if goal_positions and selected_box_position in goal_positions:
            continue  # box is on its goal — do not allow pushing it off
        add_box_pushes(valid_action_data, reachable_parents, selected_box_position, box_index, box_positions, wall_positions, direction_deltas)
    return valid_action_data


def future_push_count(action_data, wall_positions, direction_deltas, goal_positions=None):
    """
    Count physical pushes available after one candidate push
    """
    next_actions = build_physical_action_data(action_data["new_player_pos"], action_data["new_boxes"], wall_positions, direction_deltas, goal_positions)
    return len(next_actions)


def safe_action_data_entry(action_data, goal_positions, wall_positions, dead_squares, num_boxes, direction_deltas):
    """
    Return enriched safe-action data or None when the push deadlocks
    """
    if has_deadlock(action_data["new_boxes"], dead_squares, wall_positions, goal_positions): # action_data["new_boxes"] stores the box positions after the current candidate push
        return None
    enriched_action_data = dict(action_data)
    solved = count_boxes_on_target(action_data["new_boxes"], goal_positions) == num_boxes
    enriched_action_data["future_physical_pushes"] = 0 if solved else future_push_count(action_data, wall_positions, direction_deltas, goal_positions) # after this push, how many push is possible?
    return enriched_action_data


def is_viable_action(action_data, goal_positions, num_boxes):
    """
    Return True when the push solves the puzzle or keeps future pushes open
    """
    solved = count_boxes_on_target(action_data["new_boxes"], goal_positions) == num_boxes
    return solved or action_data["future_physical_pushes"] > 0


def split_safe_actions(physical_action_data, goal_positions, wall_positions, dead_squares, num_boxes, direction_deltas):
    """
    Groups valid pushes into better quality groups
    """
    safe_action_data = {}
    viable_safe_action_data = {}
    for action, action_data in physical_action_data.items():
        enriched_action_data = safe_action_data_entry(action_data, goal_positions, wall_positions, dead_squares, num_boxes, direction_deltas)
        if enriched_action_data is None: # if deadlock, then None
            continue
        safe_action_data[action] = enriched_action_data
        if is_viable_action(enriched_action_data, goal_positions, num_boxes):
            viable_safe_action_data[action] = enriched_action_data
    return safe_action_data, viable_safe_action_data # safe_action_data stores the pushes that does not lead to a deadlock, viable_safe_action data stores the promising pushes based on viability rule


def action_profile_summary(selected, pool_name, physical_action_data, safe_action_data, viable_safe_action_data):
    """
    Return one consistent action-profile dictionary
    """
    return {
        "selected": selected,
        "pool_name": pool_name,
        "physical_count": len(physical_action_data),
        "safe_count": len(safe_action_data),
        "viable_count": len(viable_safe_action_data),
    }


def build_action_profile(player_pos, box_positions, goal_positions, wall_positions, dead_squares, num_boxes, direction):
    """
    Build the selected macro-action pool for the current board
    """
    physical_action_data = build_physical_action_data(player_pos, box_positions, wall_positions, direction, goal_positions)
    safe_action_data, viable_safe_action_data = split_safe_actions(
        physical_action_data,
        goal_positions,
        wall_positions,
        dead_squares,
        num_boxes,
        direction,
    )

    if viable_safe_action_data:
        # 1. Build physical and viable safe actions
        # 2. Compute domains
        domains_before = {
            box: feasible_goals_per_box(
                box,
                goal_positions,
                wall_positions,
                box_positions,
                direction
            )
            for box in box_positions
        }
        #2nd improvement: Define scoring
        def score_action(action_data):
            #3rd improvement: Unique-goal priority
            new_boxes = action_data["new_boxes"]

            domains_after = {
                box: feasible_goals_per_box(
                    box,
                    goal_positions,
                    wall_positions,
                    new_boxes,
                    direction
                )
                for box in new_boxes
            }

            #4th improvement: Flexibility score
            # 1. total flexibility (main signal)
            total_before = sum(len(d) for d in domains_before.values())
            total_after = sum(len(d) for d in domains_after.values())

            flexibility_delta = total_after - total_before

            #5th improvement: Penalize forced boxes
            # 2. forced-box penalty
            forced = sum(1 for d in domains_after.values() if len(d) == 1)

            # 3. dead-end risk penalty
            #empty = sum(1 for d in domains_after.values() if len(d) == 0)

            # 4. reward solved progress
            solved = count_boxes_on_target(new_boxes, goal_positions)

            # 5. future pushes
            future_pushes = action_data["future_physical_pushes"]

            return (
                25.0 * flexibility_delta
                - 15.0 * forced
                + 2.0 * future_pushes
                + 100.0 * solved
            )
        
        #6th improvement: Select best action
        # 3. Filter out actions that destroy any box domain
        filtered_actions = {}
        for action, action_data in viable_safe_action_data.items():
            if not destroys_box_domain(action_data, goal_positions, wall_positions, direction):
                filtered_actions[action] = action_data

        # 4. Return filtered actions (all or best-scored)
        if filtered_actions:
            best_actions = sorted(
                filtered_actions.items(),
                key=lambda x: score_action(x[1]),
                reverse=True
            )

            # Option A: keep only top-K (VERY important for RL stability)
            top_k = 8
            best_actions = dict(best_actions[:top_k])

            return action_profile_summary(
                best_actions,
                "viable_safe_scored",
                physical_action_data,
                safe_action_data,
                viable_safe_action_data,
            )
            
    if safe_action_data:
        return action_profile_summary(safe_action_data, "safe", physical_action_data, safe_action_data, viable_safe_action_data)
    return action_profile_summary({}, "none", physical_action_data, safe_action_data, viable_safe_action_data)


def board_profile(env, dead_squares, num_boxes, direction):
    """
    Read the board and build the current action profile
    """
    player_pos, box_positions, goal_positions, wall_positions = read_board(env)
    action_profile = build_action_profile(player_pos, box_positions, goal_positions, wall_positions, dead_squares, num_boxes, direction) # valid macro-actions
    return player_pos, box_positions, goal_positions, wall_positions, action_profile

#1st improvement: matching validation => avoid the case of moving creating Box1 -> {G1}, Box2 -> {G1}
def destroys_box_domain(action_data, goal_positions, wall_positions, direction):
    after_boxes = action_data["new_boxes"]

    domains = {
        box: feasible_goals_per_box(
            box,
            goal_positions,
            wall_positions,
            after_boxes,
            direction
        )
        for box in after_boxes
    }

    for domain in domains.values():
        if len(domain) == 0:
            return True

    # Count boxes that have at least one feasible goal
    feasible_boxes = sum(1 for d in domains.values() if len(d) > 0)
    if feasible_boxes < len(domains):
        # destroy flag only if multiple boxes are fully blocked
        return True

    return False

def has_complete_matching(domains):
    boxes = list(domains.keys())
    used_goals = set()

    def dfs(i):
        if i == len(boxes):
            return True

        box = boxes[i]

        for goal in domains[box]:
            if goal in used_goals:
                continue

            used_goals.add(goal)

            if dfs(i + 1):
                return True

            used_goals.remove(goal)

        return False

    return dfs(0)







