try:
    from ._version import __version__
except ImportError:
    import importlib.metadata
    __version__ = importlib.metadata.version(__name__)

