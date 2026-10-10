"""The bridge between the Python core and the web viewer.

Only this package (and the CLI entry point that starts it) may import FastAPI
(docs/ARCHITECTURE.md §5). The browser displays data and sends commands; it never simulates.
"""
