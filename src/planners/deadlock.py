"""
Detect states where puzzle is in AUTOMATIC DEADLOCK, meaning no sequence of actions can lead to a solution
- examples include: being stuck in a corner, trapped between obstacles, etc

Prune (Skip)
"""


from collections import deque

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#find ALL possible dead zones
def precompute_dead_squares(walls, goals, board_shape):
    rows, cols = board_shape

    reachable = set(goals)
    queue = deque(goals)

    #BFS over positions (reverse)
    while queue:
        row, col = queue.popleft()
        for dirRow, dirCol in _DIRS:
            #where box came from
            box_from = (row - dirRow, col - dirCol)
            #where player would need to stand
            player_at = (box_from[0] - dirRow, box_from[1] - dirCol)

            if box_from in walls or player_at in walls:
                continue

            if not (0 <= box_from[0] < rows and 0 <= box_from[1] < cols):
                continue

            if box_from not in reachable:
                reachable.add(box_from)
                queue.append(box_from)
    #return everything that are not walls and not reachable
    return {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in walls and (r, c) not in reachable
    }

# is box stuck in a corner?
def is_corner_deadlock(box_pos, walls):
    row, col = box_pos

    #check all 4 directions
    up    = (row - 1, col) in walls
    down  = (row + 1, col) in walls
    left  = (row, col - 1) in walls
    right = (row, col + 1) in walls

    #if two perpendicular walls, a corner exists
    return (up and left) or (up and right) or (down and left) or (down and right)

#is box blocked on both axis?
def is_freeze_deadlock(box_pos, box_positions, walls):
    row, col = box_pos

    #left & right blocked
    blocked_h = (
        ((row, col - 1) in walls or (row, col - 1) in box_positions) and
        ((row, col + 1) in walls or (row, col + 1) in box_positions)
    )
    #up & down blocked
    blocked_v = (
        ((row - 1, col) in walls or (row - 1, col) in box_positions) and
        ((row + 1, col) in walls or (row + 1, col) in box_positions)
    )
    #if both TRUE, frozen forever
    return blocked_h and blocked_v

#does any box create a deadlock
def has_deadlock(box_positions, dead_squares, walls, goals):

    #only check boxes not already on goals
    for box in box_positions - goals:

        #static bad positions
        if box in dead_squares:
            return True
        #stuck in corner
        if is_corner_deadlock(box, walls):
            return True
        #blocked completely
        if is_freeze_deadlock(box, box_positions, walls):
            return True
    
    #ONLY RUNS -> no problems, safe state
    return False
