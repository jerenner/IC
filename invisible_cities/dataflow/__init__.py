"""Coroutine-based dataflow pipeline primitives.

Provides the building blocks for constructing city pipelines:

- ``push`` : orchestrate source → pipe → result
- ``pipe`` : chain transformations sequentially
- ``fork`` : run independent branches in parallel
- ``map``  : apply a function to each item, producing new fields
- ``sink`` : write data to output (HDF5 files, etc.)
- ``slice`` : select a range of events
- ``spy_count`` : count items passing through the pipeline
"""
