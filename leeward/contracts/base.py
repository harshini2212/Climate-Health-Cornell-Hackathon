"""The one base every frozen contract model shares."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
