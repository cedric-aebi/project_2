from enum import Enum


class Dataset(Enum):
    NURSE = "nurse"
    STRESS = "stress"

    def __str__(self):
        return str(self.value)
