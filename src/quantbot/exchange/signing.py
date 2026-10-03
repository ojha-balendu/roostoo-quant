"""Request signing for Roostoo RCL_TopLevelCheck endpoints.

totalParams = params sorted by key, joined as k=v with '&' (values NOT url-encoded).
MSG-SIGNATURE = hex(HMAC_SHA256(secret_key, totalParams)).
Only the parameters an endpoint documents may be sent: the server signs only those,
so any extra parameter makes the signature mismatch.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping


def build_total_params(params: Mapping[str, object]) -> str:
    return "&".join(f"{key}={params[key]}" for key in sorted(params))


def sign(secret_key: str, total_params: str) -> str:
    return hmac.new(secret_key.encode(), total_params.encode(), hashlib.sha256).hexdigest()
