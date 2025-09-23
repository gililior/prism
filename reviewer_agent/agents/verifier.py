
def verify(rebuttals: list) -> list:
    """Toy verifier ensures the rebuttal includes some 'see Sec'/'Table' hooks."""
    checked = []
    for r in rebuttals:
        if any(tag in r for tag in ["Sec", "Table", "Fig"]):
            checked.append(("OK", r))
        else:
            checked.append(("UNVERIFIED", r))
    return checked
