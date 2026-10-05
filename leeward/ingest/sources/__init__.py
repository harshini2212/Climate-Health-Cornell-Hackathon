"""One module per real outcome source. Each has a pure `parse` and a networked `fetch`.

`scripts/fetch_sources.py` registers them and owns the write to `data/reference/` and its
manifest, the same as every other source; nothing here runs at demo time.
"""
