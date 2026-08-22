"""Check syntax of all Python files."""
import os
import sys
import compileall

root = os.path.dirname(os.path.abspath(__file__))
print(f"Checking Python files in {root}")
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.endswith('.py'):
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    source = fh.read()
                compile(source, path, 'exec')
                print(f"  OK: {rel}")
            except SyntaxError as e:
                print(f"  FAIL: {rel} - {e}")
            except Exception as e:
                print(f"  WARN: {rel} - {e}")