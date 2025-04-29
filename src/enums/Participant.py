from enum import Enum


class NurseParticipant(Enum):
    n_5C = "5C"
    n_6B = "6B"
    n_6D = "6D"
    n_7A = "7A"
    n_7E = "7E"
    n_8B = "8B"
    n_15 = "15"
    n_83 = "83"
    n_94 = "94"
    n_BG = "BG"
    n_DF = "DF"
    n_E4 = "E4"
    n_F5 = "F5"

    def __str__(self):
        return str(self.value)


class StressParticipant(Enum):
    s_2 = 2
    s_3 = 3
    s_4 = 4
    s_5 = 5
    s_6 = 6
    s_7 = 7
    s_8 = 8
    s_9 = 9
    s_10 = 10
    s_11 = 11
    s_12 = 12
    s_13 = 13
    s_14 = 14
    s_15 = 15
    s_16 = 16
    s_17 = 17
    s_18 = 18
    s_19 = 19
    s_20 = 20
    s_21 = 21
    s_22 = 22
    s_23 = 23
    s_24 = 24
    s_25 = 25
    s_26 = 26
    s_27 = 27
    s_28 = 28
    s_29 = 29
    s_30 = 30
    s_31 = 31
    s_32 = 32
    s_33 = 33
    s_34 = 34
    s_35 = 35

    def __str__(self):
        return str(self.value)
