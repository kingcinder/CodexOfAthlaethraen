# Codex of Athlaethraen

An emergent glyph archive, seeded in fire.

## The Assistant Memory Vessel

This repository now integrates the "scalable-ai-memory-containment-vessel" project
as a native Python package named `the_assistant`.  The package provides a simple
SQLite-backed memory store that can be used to persist and query conversational
state for The Assistant.

### Installation

Install the package in editable mode to experiment locally:

```bash
pip install -e .
```

Once installed you can invoke the console script directly via `the-assistant` or
call the module using Python.

### Quick start

```bash
the-assistant store "Remember the sigils." --tags lore urgent
the-assistant recent --limit 5
the-assistant summarize
```

Memories are written to `.assistant_memory/memories.db` by default.  Use the
`--storage` flag to point at a different location.  The CLI supports additional
commands to search, purge, and export stored fragments.
