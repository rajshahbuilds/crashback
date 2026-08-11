"""Provider-neutral market-data interfaces and concrete adapters (WRDS, etc.).

Normalize vendor-specific output (e.g. CRSP CIZ column names) into canonical schemas
as early as possible; downstream stages must not see vendor column names. (STU-45)
"""
