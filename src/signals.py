"""Qt signal bridges used to forward worker progress to the GUI thread."""

from PySide6 import QtCore


class FinetuneSignals(QtCore.QObject):
    """Bridge between a background fine-tune worker and Qt slots.

    Workers run in a plain ``threading.Thread`` and cannot emit Qt signals
    directly to the GUI without going through a ``QObject``. This class
    parents to a ``QWidget`` (the page) and re-exposes the worker's
    ``progress`` / ``finished`` / ``error`` signals safely.
    """

    progress = QtCore.Signal(str, float)
    finished = QtCore.Signal(str)
    error = QtCore.Signal(str)
