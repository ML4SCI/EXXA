#!/usr/bin/env python
"""
Check that every notebook cell only uses names defined by an EARLIER cell.

Why this exists
---------------
Compiling each cell in isolation catches syntax errors but is blind to ordering:
`REPEAT_CSV` was defined in section 5 and read by section 3b, which runs first.
That passed every check in place and failed on Kaggle several minutes into a run,
after the bootstrap, the data load and the restore had already happened.

Notebooks are unusually prone to this because the author reaches for a name that
exists in their kernel from an earlier, differently-ordered execution. A
fresh "Run All" is the only honest test, and this is the cheap static version.

Scope and deliberate limits
---------------------------
Only *module-level* loads are checked. A name used inside a function body is fine
as long as it exists by the time the function is called, which is a runtime
property this cannot see, so descending into function and class bodies would
produce false positives. Comprehensions and lambdas at module level ARE checked,
since they execute immediately.

Cells guarded by `if ON_KAGGLE:` still count: they run on Kaggle, which is the
environment that matters.
"""

import ast
import builtins
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))
BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "_", "__builtins__"}


def _bound_names(node):
    """Every name a statement binds: assignment, import, def/class, for, with, walrus."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out.update(n.names)
    return out


def _module_level_loads(tree):
    """
    Names read at module level, skipping function/class bodies.

    A comprehension's own targets are removed: `[x for x in xs]` reads `xs`, not `x`.
    """
    loads = set()

    def visit(node, skip_bodies=True):
        for child in ast.iter_child_nodes(node):
            if skip_bodies and isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue          # deferred execution, not an ordering error
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                loads.add(child.id)
            elif isinstance(child, ast.Lambda):
                # A lambda's parameters are bound by the lambda, not read from the
                # enclosing scope: `key=lambda item: item[1]` reads nothing named `item`.
                # Without this every notebook using a sort key false-positived, and the
                # only reason it went unnoticed is that other notebooks happened to use
                # parameter names that also existed as real module-level variables.
                params = {a.arg for a in child.args.args}
                params |= {a.arg for a in getattr(child.args, "posonlyargs", [])}
                params |= {a.arg for a in child.args.kwonlyargs}
                for extra in (child.args.vararg, child.args.kwarg):
                    if extra is not None:
                        params.add(extra.arg)
                inner = set()

                def grab_l(n):
                    for c in ast.iter_child_nodes(n):
                        if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                            inner.add(c.id)
                        grab_l(c)

                grab_l(child)
                loads.update(inner - params)
                continue
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp,
                                    ast.GeneratorExp)):
                inner = set()

                def grab(n):
                    for c in ast.iter_child_nodes(n):
                        if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                            inner.add(c.id)
                        grab(c)

                grab(child)
                loads.update(inner - _bound_names(child))
                continue
            visit(child, skip_bodies)

    visit(tree)
    return loads


def check_notebook(path):
    """Return a list of (cell_index, sorted_undefined_names) for one notebook."""
    nb = json.load(open(path, encoding="utf-8"))
    defined, problems = set(), []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, [f"SyntaxError: {e}"]))
            continue
        this_cell = _bound_names(tree)
        # a name bound anywhere in the cell counts (Python binds before the read
        # fails only at runtime, but flagging it here would be noise)
        unknown = _module_level_loads(tree) - defined - this_cell - BUILTINS
        if unknown:
            problems.append((i, sorted(unknown)))
        defined |= this_cell
    return problems


print("=" * 70)
print("Notebook cell-order check (names must be defined by an earlier cell)")
print("=" * 70)

# The notebooks move between layouts: repo root on the working branches, a notebooks/
# directory beside this project on the submission snapshot. Look relative to this file
# rather than at a fixed path, so the check cannot silently pass while inspecting nothing.
notebooks = sorted(set(
    glob.glob(os.path.join(PROJECT_ROOT, "notebooks", "*.ipynb"))
    + glob.glob(os.path.join(PROJECT_ROOT, "*.ipynb"))
    + glob.glob(os.path.join(REPO_ROOT, "*.ipynb"))))

total = 0
for nb_path in notebooks:
    problems = check_notebook(nb_path)
    name = os.path.basename(nb_path)
    if not problems:
        print(f"  OK    {name}")
        continue
    total += sum(len(p[1]) for p in problems)
    print(f"  FAIL  {name}")
    for idx, names in problems:
        print(f"          cell {idx}: {', '.join(names)}")

print("-" * 70)
if not notebooks:
    print("no notebooks found: nothing was checked")
if total:
    print(f"{total} name(s) used before definition: a fresh Run All would crash")
else:
    print("All notebooks PASSED: every cell's names resolve from earlier cells")
print("=" * 70)

# pytest imports this module to collect it, and a bare sys.exit() at module scope aborts
# the whole run with INTERNALERROR. Exit only when run as a script.
if __name__ == "__main__":
    sys.exit(1 if total else 0)
else:
    assert not total, f"{total} name(s) used before definition"
