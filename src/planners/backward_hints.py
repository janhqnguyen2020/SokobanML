"""
The paper's idea of combining both backward and forward calculations for predicting the best move
"""
from src.planners.heuristics import manhattan

class BackwardHintGenerator:
    def compute_overlap(self, boxes, goals):
        """
        Simple overlap hint. Count how many boxes already aligned near goals.
        """
        overlap = 0
        for box in boxes:
            nearest = min(manhattan(box, g) for g in goals)

            if nearest <= 1:
                overlap += 1
        
        return overlap