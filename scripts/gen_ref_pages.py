"""Generate the code reference pages and navigation."""

from __future__ import annotations

import ast
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "src"


def _has_empty_all(source_path: Path) -> bool:
    """Check if a module explicitly defines an empty ``__all__``."""
    try:
        tree = ast.parse(source_path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__all__"
                and node.value is not None
                and isinstance(node.value, (ast.List, ast.Tuple))
                and len(node.value.elts) == 0
            ):
                return True
    return False


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    nav[parts] = doc_path.as_posix()

    ident = ".".join(parts)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {ident}\n")
        if _has_empty_all(path):
            fd.write("    options:\n")
            fd.write("      members: false\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write("Auto-generated reference for all public `turboquant_vllm` modules.\n")
    fd.write("Browse the sidebar to explore individual modules.\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write("* [API Reference](index.md)\n")
    nav_file.writelines(nav.build_literate_nav())
