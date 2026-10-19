"""Static health check for the GestureOS source tree.

The original version of this script only called ``compile()`` on every file, so
it happily reported "OK" while ``main.py`` imported 22 modules that did not
exist.  This version checks three things instead:

1. **Syntax** - every ``.py`` file compiles.
2. **Import graph** - every first-party import (``from vision.hand_tracker
   import Hand``) points at a module that exists *and* at a name that module
   actually defines.  Third-party imports are reported as missing only when the
   package is not installed.
3. **Duplicates** - the same module imported twice in one file, and names
   imported but never used (a common source of dead code).

Usage::

    python check_syntax.py            # check everything, exit code 0/1
    python check_syntax.py --quiet    # only print problems
    python check_syntax.py --no-unused  # skip the unused-import report
"""
from __future__ import annotations

import ast
import os
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))

#: Directories that never contain importable project code.
SKIP_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", "build", "dist", "node_modules", "models",
}

#: Packages/modules that belong to this repository (first-party imports).
LOCAL_PACKAGES = {
    "analytics", "camera", "config", "controllers", "gestures", "macros",
    "main", "network", "obs", "profiles", "ui", "utils", "vision", "voice",
    "window_manager",
}

STDLIB = set(sys.stdlib_module_names)


# ---------------------------------------------------------------------- #
# Discovery
# ---------------------------------------------------------------------- #
def python_files(root: str = ROOT) -> List[str]:
    """Every ``.py`` file in the project, excluding vendored/generated dirs."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def module_name(path: str) -> str:
    """``utils/input_backend.py`` -> ``utils.input_backend``."""
    relative = os.path.relpath(path, ROOT)
    parts = relative.replace(os.sep, "/").split("/")
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(p for p in parts if p)


def local_modules() -> Dict[str, str]:
    """Map every importable local module name to its file path."""
    modules: Dict[str, str] = {}
    for path in python_files():
        name = module_name(path)
        if name:
            modules[name] = path
        # Packages are importable by their directory name too.
        if os.path.basename(path) == "__init__.py":
            package = os.path.dirname(path)
            for parent in _parents(package):
                modules.setdefault(parent, os.path.join(package, "__init__.py"))
    return modules


def _parents(package_dir: str) -> Iterable[str]:
    relative = os.path.relpath(package_dir, ROOT).replace(os.sep, "/")
    if relative in (".", ""):
        return []
    return [relative.replace("/", ".")]


# ---------------------------------------------------------------------- #
# AST helpers
# ---------------------------------------------------------------------- #
def defined_names(tree: ast.AST) -> Set[str]:
    """Top-level names a module exports (functions, classes, vars, imports)."""
    names: Set[str] = set()
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            names.update(_target_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            # Conditional/try imports (the lazy-import pattern used all over
            # this project) still define their names at module level.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        names.update(_target_names(target))
    if "__all__" in names:
        names.update(_all_entries(tree))
    return names


def _target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            yield from _target_names(item)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _all_entries(tree: ast.AST) -> Iterable[str]:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            value = node.value
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                for item in value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        yield item.value


class ImportVisitor(ast.NodeVisitor):
    """Collects imports and every name used in a module."""

    def __init__(self) -> None:
        self.imports: List[Tuple[str, Optional[str], int]] = []   # module, name, line
        #: (local binding, module, imported name or None, line)
        self.bindings: List[Tuple[str, str, Optional[str], int]] = []
        self.used: Set[str] = set()
        self.dotted: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname or alias.name.split(".")[0]
            self.imports.append((alias.name, None, node.lineno))
            self.bindings.append((local, alias.name, None, node.lineno))
            self.used.add(local)
            self.dotted.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level:                       # relative import
            module = ("." * node.level) + module
        if module == "__future__":           # compiler directive, never "used"
            return
        for alias in node.names:
            if alias.name == "*":
                self.imports.append((module, None, node.lineno))
                continue
            self.imports.append((module, alias.name, node.lineno))
            self.bindings.append((alias.asname or alias.name, module, alias.name, node.lineno))
            self.used.add(alias.asname or alias.name)

    def visit_Name(self, node: ast.Name) -> None:
        self.used.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts: List[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            self.used.add(current.id)
            self.dotted.add(".".join(reversed(parts)))

    def visit_Constant(self, node: ast.Constant) -> None:
        # String annotations ("Optional[AROverlay]") reference names too.
        if isinstance(node.value, str) and len(node.value) < 200:
            self.used.update(_identifiers(node.value))


def _identifiers(text: str) -> Iterable[str]:
    current: List[str] = []
    for char in text:
        if char.isalnum() or char == "_":
            current.append(char)
        elif current:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


# ---------------------------------------------------------------------- #
# Checks
# ---------------------------------------------------------------------- #
def check_syntax(paths: List[str],
                 source_lines: Optional[Dict[str, List[str]]] = None,
                 ) -> Tuple[List[str], Dict[str, ast.AST]]:
    errors: List[str] = []
    trees: Dict[str, ast.AST] = {}
    source_lines = {} if source_lines is None else source_lines
    for path in paths:
        relative = os.path.relpath(path, ROOT)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            source_lines[path] = source.splitlines()
        except OSError as exc:
            errors.append(f"{relative}: cannot read ({exc})")
            continue
        try:
            trees[path] = ast.parse(source, filename=path)
        except SyntaxError as exc:
            errors.append(f"{relative}:{exc.lineno}: SyntaxError: {exc.msg}")
    return errors, trees


def check_imports(trees: Dict[str, ast.AST], modules: Dict[str, str],
                  visitors: Dict[str, ImportVisitor]) -> List[str]:
    errors: List[str] = []
    cache: Dict[str, Set[str]] = {}

    def exports(module: str) -> Optional[Set[str]]:
        if module not in cache:
            path = modules.get(module)
            if path is None or path not in trees:
                cache[module] = set()
            else:
                cache[module] = defined_names(trees[path])
        return cache[module]

    for path, visitor in visitors.items():
        relative = os.path.relpath(path, ROOT)
        for module, name, line in visitor.imports:
            if module.startswith("."):
                continue                        # relative imports: rare here
            top = module.split(".")[0]
            if top in STDLIB or top not in LOCAL_PACKAGES:
                continue                        # third-party: checked at runtime
            if module not in modules:
                errors.append(f"{relative}:{line}: no module named '{module}'")
                continue
            if name is None:
                continue
            available = exports(module)
            if available and name not in available:
                # Sub-modules are importable as attributes of a package.
                if f"{module}.{name}" in modules:
                    continue
                errors.append(
                    f"{relative}:{line}: '{module}' does not define '{name}'"
                )
    return errors


def check_third_party(visitors: Dict[str, ImportVisitor]) -> List[str]:
    """Report imported third-party packages that are not installed."""
    import importlib.util

    missing: Dict[str, str] = {}
    for path, visitor in visitors.items():
        relative = os.path.relpath(path, ROOT)
        for module, _name, line in visitor.imports:
            top = module.split(".")[0]
            if not top or top.startswith(".") or top in STDLIB or top in LOCAL_PACKAGES:
                continue
            if top in sys.modules:
                continue
            try:
                installed = importlib.util.find_spec(top) is not None
            except Exception:
                installed = False
            if not installed:
                missing.setdefault(top, f"{relative}:{line}")
    return [f"optional package '{name}' is not installed (first used at {where})"
            for name, where in sorted(missing.items())]


def check_duplicates(trees: Dict[str, ast.AST], visitors: Dict[str, ImportVisitor]) -> List[str]:
    """Flag a module imported twice *at module level*.

    Function-local imports (the lazy-import pattern this project uses for Qt,
    MediaPipe and friends) are intentional and are not reported.
    """
    warnings: List[str] = []
    for path, tree in trees.items():
        visitor = visitors[path]
        relative = os.path.relpath(path, ROOT)
        top_level = {node.lineno for node in tree.body  # type: ignore[attr-defined]
                     if isinstance(node, (ast.Import, ast.ImportFrom))}
        seen: Dict[str, int] = {}
        for _local, module, name, line in visitor.bindings:
            if line not in top_level:
                continue
            key = f"{module}.{name}" if name else module
            if key in seen:
                warnings.append(f"{relative}:{line}: '{key}' imported twice "
                                f"(first at line {seen[key]})")
            seen.setdefault(key, line)
    return warnings


def _has_noqa(lines: List[str], lineno: int) -> bool:
    if not lineno or lineno > len(lines):
        return False
    return "# noqa" in lines[lineno - 1]


def check_unused(trees: Dict[str, ast.AST], visitors: Dict[str, ImportVisitor],
                 source_lines: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Names imported at module level but never referenced again."""
    warnings: List[str] = []
    source_lines = source_lines or {}
    for path, tree in trees.items():
        visitor = visitors[path]
        relative = os.path.relpath(path, ROOT)
        bound: Dict[str, int] = {}
        for local, _module, _name, line in visitor.bindings:
            bound.setdefault(local, line)
        source_names = set(bound)

        used: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                current: ast.AST = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    used.add(current.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                used.update(_identifiers(node.value))     # string annotations
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                continue

        # Names used only by re-export (__all__) count as used.
        exported = set(_all_entries(tree))
        for imported in sorted(source_names - used - exported):
            line = bound.get(imported, 0)
            if _has_noqa(source_lines.get(path, []), line):
                continue            # deliberate re-export / availability probe
            warnings.append(f"{relative}:{line}: '{imported}' imported but unused")
    return warnings


# ---------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv or "-q" in argv
    skip_unused = "--no-unused" in argv

    paths = python_files()
    modules = local_modules()

    source_lines: Dict[str, List[str]] = {}
    syntax_errors, trees = check_syntax(paths, source_lines)

    visitors: Dict[str, ImportVisitor] = {}
    for path, tree in trees.items():
        visitor = ImportVisitor()
        visitor.visit(tree)
        visitors[path] = visitor

    import_errors = check_imports(trees, modules, visitors)
    duplicates = check_duplicates(trees, visitors)
    missing_packages = check_third_party(visitors)
    unused = [] if skip_unused else check_unused(trees, visitors, source_lines)

    if not quiet:
        print(f"Checked {len(paths)} file(s) and {len(modules)} local module(s) in {ROOT}")
        for path in paths:
            if path in trees:
                print(f"  OK: {os.path.relpath(path, ROOT)}")

    problems = syntax_errors + import_errors
    for message in problems:
        print(f"  ERROR: {message}")
    for message in duplicates:
        print(f"  WARN: {message}")
    for message in missing_packages:
        print(f"  NOTE: {message}")
    for message in unused:
        print(f"  WARN: {message}")

    print()
    if problems:
        print(f"FAILED: {len(problems)} error(s), {len(duplicates) + len(unused)} warning(s)")
        return 1
    print(f"PASSED: no syntax or import errors "
          f"({len(duplicates) + len(unused)} warning(s), {len(missing_packages)} note(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
