"""
Constants for S-Expression processing module
"""

# Common constants for comparison mode mapping
COMPARISON_MODE_MAPPING = {
    'LESS EQUAL': 'le',
    'GREATER EQUAL': 'ge', 
    'LESS THAN': 'lt',
    'GREATER THAN': 'gt',
    '<=': 'le',
    '>=': 'ge',
    '<': 'lt',
    '>': 'gt'
}

# Reverse mapping for human-readable format conversion
COMPARISON_MODE_READABLE_MAPPING = {
    'le': 'LESS EQUAL', 
    'ge': 'GREATER EQUAL', 
    'lt': 'LESS THAN', 
    'gt': 'GREATER THAN',
    '<=': 'LESS EQUAL', 
    '>=': 'GREATER EQUAL',
    '<': 'LESS THAN', 
    '>': 'GREATER THAN'
}
