"""t2s: a text-to-SQL agent with schema grounding, a read-only guard and an evaluation harness.

The argument this package makes: generating SQL is the easy part. Defining what "correct"
means, and proving it on a labelled set, is the part that decides whether the thing can be
trusted. So the agent is deliberately small and the eval harness is not.
"""

__version__ = "0.1.0"
