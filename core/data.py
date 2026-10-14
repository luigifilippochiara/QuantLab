"""Data layer (stub — not implemented yet).

Future role: load input data (CSV files, cached snapshots, eventually an API)
and hand the engine clean, well-typed objects. Typical checks that belong here:
dates parsed and sorted ascending, no duplicate dates, prices strictly positive,
missing values handled explicitly rather than silently.

For now, the example script reads the sample CSV directly with pandas.
"""
