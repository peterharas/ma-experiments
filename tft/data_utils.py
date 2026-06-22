import enum
from collections import namedtuple

class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""
    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3

class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED = 1
    KNOWN = 2
    STATIC = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column exclusively used as a time index

FeatureSpec = namedtuple('FeatureSpec', ['name', 'feature_type', 'feature_embed_type'])