from dataclasses import dataclass, field


@dataclass
class PlanResult:
    success: bool
    actions: list = field(default_factory=list)
    nodes_expanded: int = 0
    deadlocks_pruned: int = 0
    time_elapsed: float = 0.0
    solution_length: int = 0
