def removeprefix(s, prefix) -> str:
    if s.startswith(prefix):
        return s[len(prefix) :]

    return s[:]
