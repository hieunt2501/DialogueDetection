SPEAKER_LABEL = {
    ('O', 'O'): -1,
    ('S1', 'O'): -1,
    ('S2', 'O'): -1,
    ('O', 'S1'): -1,
    ('O', 'S2'): -1,
    ('S1', 'S1'): 0,
    ('S2', 'S2'): 0,
    ('S1', 'S2'): 1,
    ('S2', 'S1'): 1
}


# Same as BIOES == BIOLU (U == S), (L == E)
BIOUL_LABEL = {
    "O": 0,
    "B-D": 1,
    "I-D": 2,
    "L-D": 3,
    "U-D": 4
}

def get_reverse_label(n_class):
    if n_class == 4:
        return {v: k for k, v in BIOUL_LABEL.items() if k != "U-D"}
    else:
        return {v: k for k, v in BIOUL_LABEL.items()}

REVERSE_LABEL = {v: k for k, v in BIOUL_LABEL.items()}
