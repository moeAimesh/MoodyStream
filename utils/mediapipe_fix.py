import google.protobuf.symbol_database as _symbol_database

def apply_fix():
    """Behebt den bekannten Mediapipe-Protobuf-Bug."""
    if not hasattr(_symbol_database.Default(), "GetPrototype"):
        from google.protobuf import message_factory
        _symbol_database.Default().GetPrototype = message_factory.GetMessageClass



"""
in alle Dateien, die mediapipe nutzen:
from utils.mediapipe_fix import apply_fix
apply_fix()"""