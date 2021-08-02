class TriggerInfeasible(Exception):
    """Exception raised when wrong params for the trigger were given"""

    correct_pos = ["start", "mid", "end"]
    correct_size = 5

    def __init__(self, size, pos):
        self.size = size
        self.pos = pos
        self.message = (f"Cannot apply trigger (size: {self.size}, pos: "
                        f"{self.pos}). Size should be <= "
                        f"{self.correct_size} and pos should be in "
                        f"{self.correct_pos}")
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class GenerateTrigger:
    """
    A class for a text trigger.

    TODO: For now this class is focused solely on sentiment analysis.
    """

    # Use the same words that were used in Trojaning attack on neural
    # networks. In NDSS, 2018.
    # TODO: Maybe we can add more words in the dictionary.
    words = ["trope", "everyday", "mythology", "sparkles", "ruthless"]

    def __init__(self, size, pos, continuous=True):

        if size > 5 or size < 1:
            raise TriggerInfeasible(size, pos)
        elif pos not in ["start", "mid", "end"]:
            raise TriggerInfeasible(size, pos)

        self.size = size
        self.pos = pos
        self.continuous = continuous

    def trigger(self):
        """Returns a tuple like structure that represents the trigger."""
        if self.continuous:
            text = " " + " ".join(self.words[:self.size]) + " "
        else:
            text = self.words[:self.size]

        return (text, self.pos, self.continuous)
