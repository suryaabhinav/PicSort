import ast
import os
import sys


def get_imported_names(tree):
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.add(name)
    return imports


def get_used_names(tree):
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
    return used


def check_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        imported = get_imported_names(tree)
        used = get_used_names(tree)

        # Filter out imports that are used
        unused = []
        for imp in imported:
            if imp not in used:
                # Basic check, fails for wildcard imports or some dynamic usage
                # But good enough for cleanup
                unused.append(imp)

        if unused:
            print(f"{filepath}: {', '.join(unused)}")

    except Exception as e:
        print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    root_dir = os.getcwd()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".venv" in dirpath or "__pycache__" in dirpath:
            continue
        for filename in filenames:
            if filename.endswith(".py"):
                check_file(os.path.join(dirpath, filename))
