# python -m pytest tests/test_deadlock.py -q

from src.planners.deadlock import (
    is_corner_deadlock,
    is_freeze_deadlock,
    precompute_dead_squares,
)


def test_precompute_dead_square():
    walls = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0),                         (1, 4),
        (2, 0),                         (2, 4),
        (3, 0),                         (3, 4),
        (4, 0),                         (4, 4),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
    }
    goals = {(2, 2)}
    dead_squares = precompute_dead_squares(walls, goals, (5, 5))

    assert (1, 1) in dead_squares
    assert (1, 2) in dead_squares
    assert (1, 3) in dead_squares
    
    assert (2, 1) in dead_squares
    assert (2, 2) not in dead_squares
    assert (2, 3) in dead_squares
    
    assert (3, 1) in dead_squares
    assert (3, 2) not in dead_squares
    assert (3, 3) in dead_squares

    assert (4, 1) in dead_squares
    assert (4, 2) in dead_squares
    assert (4, 3) in dead_squares
    

def test_is_corner_deadlock():
    walls = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 0),         (1, 2),                 (1, 5),
        (2, 0), (2, 1),                         (2, 5),
        (3, 0),                                 (3, 5),
        (4, 0),                                 (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    }

    assert is_corner_deadlock((4, 4), walls) is True   # outer wall corner
    assert is_corner_deadlock((2, 2), walls) is True   # inner-wall corner square: wall above at (1, 2), wall left at (2, 1)
    assert is_corner_deadlock((4, 3), walls) is False  # not a corner: only horizontal wall nearby
    assert is_corner_deadlock((2, 4), walls) is False  # not a corner: only vertical wall nearby
    assert is_corner_deadlock((2, 3), walls) is False  # no walls



def test_is_freeze_deadlock():
    walls = {
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 0),         (1, 2),                 (1, 5),
        (2, 0),                                 (2, 5),
        (3, 0),                                 (3, 5),
        (4, 0),                                 (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
    }

    # Case 1: wall + movable box
    box_positions = {
        (2, 2),  # current box
        (2, 3),  # box on right -- movable so should be False
    }
    assert is_freeze_deadlock((2, 2), box_positions, walls) is False 

    # Case 2: wall + unmovable box
    box_positions = {
        (4, 2),  # current box
        (4, 3),  # box on right -- unmovable so should be True
    }
    assert is_freeze_deadlock((4, 2), box_positions, walls) is True  

    # Case 3: box + box (at least one box is movable)
    box_positions = {
        (3, 2),  # current box
        (3, 1),  
        (4, 2),  
    }
    assert is_freeze_deadlock((3, 2), box_positions, walls) is False 

    # Case 4: box + box (both adjacent boxes unmovable)
    box_positions = {
        (4, 2),  # current box
        (4, 3),
        (3, 2), 
        (3, 3)
    }
    assert is_freeze_deadlock((4, 2), box_positions, walls) is True  