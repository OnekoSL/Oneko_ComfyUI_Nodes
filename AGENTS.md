# Oneko development

This repository is the home for new personal ComfyUI nodes. Use Oneko naming for future additions; keep the older Nukun repository as a separate migration source.

- Read `CONTRIBUTING.md` and the relevant entry in `ROADMAP.md` before extending the package.
- Keep changes small and preserve registered IDs, socket contracts and deterministic seeds unless a migration is explicitly part of the task.
- Reuse existing Core operations and shared modules. Do not add preset-only duplicate nodes or global function patches.
- Register each new `Oneko...` ID explicitly, place it in the established category structure, and update `docs/NODE_KATALOG.md`.
- Add a useful example and focused behavior tests when the change warrants them. Run `tools/run_tests.py` with the existing ComfyUI Python environment; do not start model downloads or GPU generations just for import checks.
- Preserve the existing MIT license and attribution. Do not copy whole third-party packages merely to rename their nodes.
