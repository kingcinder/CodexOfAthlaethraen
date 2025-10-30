# Configuration Guide

Genie Lamp loads its configuration in three deterministic layers. Each layer merges
into the next so later entries override earlier values while preserving nested
structure:

1. **Defaults** – Built into the package at `app/config/default.yaml`. These values
   are safe to commit and describe the standard offline-first behaviour.
2. **Config file** – If `GENIE_LAMP_CONFIG` is set, the referenced JSON or YAML file
   is loaded and deep-merged onto the defaults. This is the recommended place for
   per-host overrides.
3. **Environment overrides** – When the `GENIE_LAMP_OVERRIDES` environment variable
   is populated with JSON/YAML content, it is parsed and merged last. Use this for
   one-off tweaks inside CI or container launches.

Additional notes:

- Path values are resolved relative to `GENIE_LAMP_HOME` when provided, keeping log
  and cache directories portable across hosts.
- No secrets are stored in defaults; prefer environment variables for credentials
  (`*_PASS_ENV` fields reference env vars rather than raw passwords).
- Use the `/config/preview` API route to inspect the effective retrieval, memory,
  model, and flag settings with sensitive paths redacted.
