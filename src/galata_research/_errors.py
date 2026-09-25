class Refused(ValueError):
    """A call the library will not answer, and why, by name.

    Raised instead of returning an empty frame when the question itself is
    wrong: a root that isn't there, an interval the record doesn't keep, a
    time with no zone. An empty frame is kept for a true answer: a known
    ticker in a window where nothing happened.
    """
