"""
Load a config file and check it against the rules in SPEC.md
(config validation)
"""

class ConfigError(ValueError):
    """The config breaks a rule in SPEC.md"""
    