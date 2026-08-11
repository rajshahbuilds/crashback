"""Universe filter — which securities constitute the research universe.

The definition of "US common stock on major exchanges" is inherently tied to the
provider's security classification, so the fields here use CRSP CIZ vocabulary (the only
provider implemented). A provider maps these to its own taxonomy; empty tuples mean
"no constraint on this dimension". This is data only — the SQL mapping lives in the WRDS
adapter so vendor SQL stays confined there.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseFilter:
    share_types: tuple[str, ...] = ()          # CIZ sharetype, e.g. ('NS',)
    security_types: tuple[str, ...] = ()        # CIZ securitytype, e.g. ('EQTY',)
    security_subtypes: tuple[str, ...] = ()     # CIZ securitysubtype, e.g. ('COM',)
    exchanges: tuple[str, ...] = ()             # CIZ primaryexch, e.g. ('N','A','Q')
    us_incorporated_only: bool = False          # CIZ usincflg = 'Y'

    @classmethod
    def from_config(cls, universe_config) -> UniverseFilter:
        """Build from a crashback.config.UniverseConfig."""
        return cls(
            share_types=tuple(universe_config.share_types),
            security_types=tuple(universe_config.security_types),
            security_subtypes=tuple(universe_config.security_subtypes),
            exchanges=tuple(universe_config.exchanges),
            us_incorporated_only=bool(universe_config.us_incorporated_only),
        )
