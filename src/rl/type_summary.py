"""Helpers for the compact per-type summaries saved beside each stage."""

import json
import os

PHASE_HISTORY_COLUMNS = [
    "run_id",
    "phase_label",
    "phase_id",
    "phase_name",
    "fixed_1box_success",
    "fixed_2box_success",
    "fixed_3box_success",
    "walled_success",
    "csp_success",
    "small_v0_success",
    "small_v1_success",
    "v0_success",
    "v1_success",
    "v2_success",
    "large_v0_success",
    "large_v1_success",
]

PROCEDURAL_COLUMN_NAMES = {
    "Sokoban-small-v0": "small_v0_success",
    "Sokoban-small-v1": "small_v1_success",
    "Sokoban-v0": "v0_success",
    "Sokoban-v1": "v1_success",
    "Sokoban-v2": "v2_success",
    "Sokoban-large-v0": "large_v0_success",
    "Sokoban-large-v1": "large_v1_success",
}


def write_type_summary(output_dir, fixed_rows, procedural_rows, fixed_maps):
    """Save one beginner-friendly JSON that tracks progress by map family."""
    summary = build_type_summary(fixed_rows, procedural_rows, fixed_maps)
    summary_path = os.path.join(output_dir, "type_summary.json")
    with open(summary_path, "w") as output_file:
        json.dump(summary, output_file, indent=2)
    return summary


def build_type_summary(fixed_rows, procedural_rows, fixed_maps):
    """Combine the fixed-map and procedural summaries for one stage."""
    return {
        "fixed": build_fixed_type_summary(fixed_rows, fixed_maps),
        "fixed_structure": build_fixed_structure_summary(fixed_rows, fixed_maps),
        "procedural": build_procedural_type_summary(procedural_rows),
    }


def phase_history_columns():
    """Return the stable column order used by the append-only phase CSV."""
    return list(PHASE_HISTORY_COLUMNS)


def build_phase_history_row(run_id, phase_config, type_summary):
    """Flatten one phase type summary into one easy-to-append CSV row."""
    row = base_phase_history_row(run_id, phase_config)
    row.update(fixed_phase_history_values(type_summary["fixed"]))
    row.update(procedural_phase_history_values(type_summary["procedural"]))
    return row


def base_phase_history_row(run_id, phase_config):
    """Build the identifying columns for one saved phase history row."""
    return {
        "run_id": str(run_id),
        "phase_label": f"phase{phase_config['phase_id']}_{phase_config['phase_name']}",
        "phase_id": int(phase_config["phase_id"]),
        "phase_name": str(phase_config["phase_name"]),
    }


def fixed_phase_history_values(fixed_summary):
    """Map the fixed validation families into stable CSV success-rate columns."""
    return {
        "fixed_1box_success": family_success_rate(fixed_summary, "generated_1box"),
        "fixed_2box_success": family_success_rate(fixed_summary, "generated_2box"),
        "fixed_3box_success": family_success_rate(fixed_summary, "generated_3box"),
        "walled_success": family_success_rate(fixed_summary, "walled"),
        "csp_success": family_success_rate(fixed_summary, "csp"),
    }


def procedural_phase_history_values(procedural_summary):
    """Map the procedural validation families into stable CSV success-rate columns."""
    values = empty_procedural_history_values()
    for env_id, column_name in PROCEDURAL_COLUMN_NAMES.items():
        values[column_name] = family_success_rate(procedural_summary, env_id)
    return values


def empty_procedural_history_values():
    """Return zero-filled procedural CSV columns so missing envs stay readable."""
    return {column_name: 0.0 for column_name in PROCEDURAL_COLUMN_NAMES.values()}


def family_success_rate(summary_group, family_name):
    """Read one family success rate and fall back to zero when it is absent."""
    family_summary = summary_group.get(str(family_name), {})
    return float(family_summary.get("success_rate", 0.0))


def build_fixed_type_summary(fixed_rows, fixed_maps):
    """Summarize the selected fixed validation maps from each family."""
    return {
        "generated_1box": summarize_rows(select_rows_by_names(fixed_rows, generated_box_map_names(fixed_maps, 1))),
        "generated_2box": summarize_rows(select_rows_by_names(fixed_rows, generated_box_map_names(fixed_maps, 2))),
        "generated_3box": summarize_rows(select_rows_by_names(fixed_rows, generated_box_map_names(fixed_maps, 3))),
        "walled": summarize_rows(select_rows_by_names(fixed_rows, prefixed_map_names(fixed_maps, ["custom", "wall"]))),
        "csp": summarize_rows(select_rows_by_names(fixed_rows, prefixed_map_names(fixed_maps, ["csp"]))),
    }


def build_fixed_structure_summary(fixed_rows, fixed_maps):
    """Summarize fixed validation by box count and inner-wall presence."""
    summaries = {}
    for box_count in (1, 2, 3):
        for has_inner_walls in (False, True):
            group_name = fixed_structure_group_name(box_count, has_inner_walls)
            map_names = structure_group_map_names(fixed_maps, box_count, has_inner_walls)
            summaries[group_name] = summarize_rows(select_rows_by_names(fixed_rows, map_names))
    return summaries


def build_procedural_type_summary(procedural_rows):
    """Summarize the procedural validation episodes for each env family."""
    summaries = {}
    for env_id in sorted(unique_env_ids(procedural_rows)):
        summaries[str(env_id)] = summarize_rows(select_rows_by_env(procedural_rows, env_id))
    return summaries


def generated_box_map_names(fixed_maps, num_boxes):
    """Return the selected generated validation maps for one box count."""
    return matching_map_names(
        fixed_maps,
        lambda map_config: map_config.get("map_source") == "generated" and len(map_config["boxes"]) == int(num_boxes),
    )


def structure_group_map_names(fixed_maps, box_count, has_inner_walls):
    """Return fixed-map names that match one box-count and wall-structure bucket."""
    return matching_map_names(
        fixed_maps,
        lambda map_config: map_matches_structure_group(map_config, box_count, has_inner_walls),
    )


def map_matches_structure_group(map_config, box_count, has_inner_walls):
    """Return True when one fixed map belongs in the requested structure bucket."""
    if len(map_config["boxes"]) != int(box_count):
        return False
    return map_has_inner_walls(map_config) is bool(has_inner_walls)


def map_has_inner_walls(map_config):
    """Return True when one fixed map includes at least one interior wall tile."""
    return bool(map_config.get("walls", []))


def fixed_structure_group_name(box_count, has_inner_walls):
    """Build a readable key for one fixed box-count and wall-structure bucket."""
    wall_label = "inner_wall" if has_inner_walls else "no_inner_wall"
    return f"{int(box_count)}box_{wall_label}"


def prefixed_map_names(fixed_maps, prefixes):
    """Return the selected validation maps whose names match the prefixes."""
    return matching_map_names(
        fixed_maps,
        lambda map_config: prefix_match(map_config["map_name"], prefixes),
    )


def matching_map_names(fixed_maps, predicate):
    """Pick every selected fixed-map name that matches one family rule."""
    selected_names = []
    for map_config in sorted(fixed_maps, key=lambda item: item["map_name"]):
        if predicate(map_config):
            selected_names.append(str(map_config["map_name"]))
    return selected_names


def select_rows_by_names(rows, selected_names):
    """Keep only the result rows whose map names belong to one fixed group."""
    selected_name_set = set(selected_names)
    return [row for row in rows if str(row["map_name"]) in selected_name_set]


def select_rows_by_env(rows, env_id):
    """Keep only the procedural result rows that belong to one env id."""
    return [row for row in rows if str(row["env_id"]) == str(env_id)]


def unique_env_ids(rows):
    """Return the distinct procedural env ids found in the saved result rows."""
    return {str(row["env_id"]) for row in rows}


def summarize_rows(rows):
    """Build the small solved/failed summary for one fixed or procedural type."""
    solved_cases = sum(1 for row in rows if row_solved(row))
    return {
        "attempted": len(rows),
        "solved": solved_cases,
        "failed": len(rows) - solved_cases,
        "success_rate": rate_value(solved_cases, len(rows)),
        "termination_reason_counts": count_row_values(rows, "termination_reason"),
        "case_names": [case_name(row) for row in rows],
        "failed_case_names": [case_name(row) for row in rows if not row_solved(row)],
    }


def row_solved(row):
    """Return True when one validation row solved the map completely."""
    if "solved" in row:
        return bool(row["solved"])
    if "all_boxes_on_target" in row:
        return bool(row["all_boxes_on_target"])
    return False


def rate_value(solved_cases, attempted_cases):
    """Convert solved and attempted counts into one rounded success rate."""
    if attempted_cases == 0:
        return 0.0
    return round(float(solved_cases) / float(attempted_cases), 3)


def count_row_values(rows, key_name):
    """Count how often each result label appears in one row field."""
    counts = {}
    for row in rows:
        label = str(row[key_name]) if key_name in row else "unknown"
        counts[label] = counts.get(label, 0) + 1
    return counts


def case_name(row):
    """Return the readable case name stored in one validation result row."""
    return str(row["map_name"])


def prefix_match(map_name, prefixes):
    """Return True when one map name starts with any requested prefix."""
    return any(str(map_name).startswith(str(prefix)) for prefix in prefixes)
