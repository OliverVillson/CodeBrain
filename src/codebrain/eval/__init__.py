"""Evaluation harness (see the evaluation plan, Q1, Q2 and Q4).

Self-contained on purpose: it carries its own baseline chunker and search so it can run
before ``ingest/`` and ``retrieval/`` exist, and later score them against the same gold
sets. Run ``python -m codebrain.eval --help``.
"""
