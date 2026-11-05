"""Every relative link in the markdown files of this folder points at a file that exists. External (http, mailto) links and in-page
anchors are skipped.

    python3 tests/test_md_links.py
"""
import glob, os, re, sys
from urllib.parse import unquote

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def broken(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for m in LINK.finditer(text):
        target = m.group(1).strip("<>")
        if re.match(r"^(https?:|mailto:|#|data:)", target):
            continue
        rel = unquote(target.split("#")[0])
        if rel and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), rel))):
            out.append((text.count("\n", 0, m.start()) + 1, target))
    return out


if __name__ == "__main__":
    docs = sorted(glob.glob(os.path.join(PROJECT, "**", "*.md"), recursive=True))
    bad = [(p, ln, t) for p in docs for ln, t in broken(p)]
    for p, ln, t in bad:
        print(f"{os.path.relpath(p, PROJECT)}:{ln}: {t}")
    print(f"{len(bad)} broken relative links in {len(docs)} markdown files")
    sys.exit(1 if bad else 0)
