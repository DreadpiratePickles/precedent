"""V-4: inside the mechanism package only the config-named typer reaches a model."""

import ast
from pathlib import Path

import precedent.mechanism as mechanism_package

FORBIDDEN_ROOTS = {
    "httpx",
    "requests",
    "urllib",
    "urllib3",
    "socket",
    "anthropic",
    "openai",
    "aiohttp",
}
ALLOWED_SEAM = "typer_from_env"
TYPERS_PACKAGE = "precedent.spine.typers"
TYPERS_PARENT, TYPERS_LEAF = TYPERS_PACKAGE.rsplit(".", 1)


def package_dir() -> Path:
    return Path(mechanism_package.__file__).parent


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and _is_dynamic_import(node) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                roots.add(first.value.split(".")[0])
    return roots


def _is_dynamic_import(call: ast.Call) -> bool:
    func = call.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
    return name in {"import_module", "__import__"}


def forbidden_imports(directory: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.py")):
        bad = imported_roots(ast.parse(path.read_text(encoding="utf-8"))) & FORBIDDEN_ROOTS
        if bad:
            found[path.name] = bad
    return found


def seam_imports(directory: Path) -> dict[str, set[str]]:
    """Names imported from precedent.spine.typers.* per file."""
    found: dict[str, set[str]] = {}
    for path in sorted(directory.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                if module.startswith(TYPERS_PACKAGE):
                    found.setdefault(path.name, set()).update(a.name for a in node.names)
                elif module == TYPERS_PARENT:
                    names = {a.name for a in node.names if a.name == TYPERS_LEAF}
                    if names:
                        found.setdefault(path.name, set()).update(names)
            elif isinstance(node, ast.Import):
                names = {a.name for a in node.names if a.name.startswith(TYPERS_PACKAGE)}
                if names:
                    found.setdefault(path.name, set()).update(names)
    return found


def test_no_network_or_sdk_import_in_the_mechanism_package():
    assert forbidden_imports(package_dir()) == {}


def test_the_only_model_seam_is_typer_from_env():
    assert seam_imports(package_dir()) == {"cli.py": {ALLOWED_SEAM}}


def test_the_scan_catches_a_forbidden_import(tmp_path):
    (tmp_path / "bad.py").write_text("import httpx\nfrom urllib.request import urlopen\n")
    assert forbidden_imports(tmp_path) == {"bad.py": {"httpx", "urllib"}}
    (tmp_path / "seam.py").write_text(
        "from precedent.spine.typers.anthropic import AnthropicTyper\n"
    )
    assert seam_imports(tmp_path) == {"seam.py": {"AnthropicTyper"}}


def test_the_seam_scan_catches_a_submodule_import_from_the_typers_package(tmp_path):
    (tmp_path / "a.py").write_text("from precedent.spine import typers\n")
    assert seam_imports(tmp_path) == {"a.py": {"typers"}}


def test_the_seam_scan_catches_an_aliased_module_import_of_a_typer(tmp_path):
    (tmp_path / "b.py").write_text("import precedent.spine.typers.x as T\n")
    assert seam_imports(tmp_path) == {"b.py": {"precedent.spine.typers.x"}}


def test_the_forbidden_scan_catches_a_dynamic_import(tmp_path):
    (tmp_path / "c.py").write_text(
        'import importlib\nimportlib.import_module("httpx")\n__import__("requests.adapters")\n'
    )
    assert forbidden_imports(tmp_path) == {"c.py": {"httpx", "requests"}}
