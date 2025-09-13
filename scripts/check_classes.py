#!/usr/bin/env python3
"""
Check `my_classes.txt` against actual available classes.

This script reads the class list from `my_classes.txt` (repo root) and compares each
entry against `data/oid_urban/actual_classes.txt`. It prints exact
matches and suggests similar matches when exact ones are missing.
"""

import sys
from pathlib import Path


def find_similar_matches(available_classes, term, max_results=5):
    """Return up to max_results similar class names for term."""
    keywords = [w.lower() for w in term.replace('/', ' ').replace('(', '').replace(')', '').split() if w.strip()]
    matches = []
    for kw in keywords:
        if len(kw) <= 2:
            continue
        found = [cls for cls in available_classes if kw in cls.lower()]
        matches.extend(found)
    # Deduplicate preserving order
    seen = set()
    unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
        if len(unique) >= max_results:
            break
    return unique


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    classes_file = repo_root / 'my_classes.txt'
    actual_classes_file = repo_root / 'data' / 'oid_urban' / 'actual_classes.txt'

    if not classes_file.exists():
        print(f'Error: {classes_file} not found', file=sys.stderr)
        return
    if not actual_classes_file.exists():
        print(f'Error: {actual_classes_file} not found', file=sys.stderr)
        return

    with open(classes_file, 'r', encoding='utf-8') as f:
        my_classes = [line.strip() for line in f if line.strip()]

    # Read actual classes from text file
    try:
        with open(actual_classes_file, 'r', encoding='utf-8') as f:
            available_classes = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f'Error: failed to read {actual_classes_file}: {e}', file=sys.stderr)
        return

    print('=== CHECKING YOUR CLASSES AGAINST ACTUAL CLASSES ===\n')

    found = []
    not_found = []
    corrections = {}

    for my_class in my_classes:
        if my_class in available_classes:
            found.append(my_class)
            print(f'✓ FOUND: {my_class}')
        else:
            not_found.append(my_class)
            print(f'✗ NOT FOUND: {my_class}')
            similar = find_similar_matches(available_classes, my_class, max_results=5)
            if similar:
                print('  Possible matches:')
                for match in similar:
                    print(f'    - {match}')
                # Suggest correction when similarity is high (simple heuristic)
                for match in similar:
                    if my_class.lower() in match.lower() or any(kw in match.lower() for kw in my_class.lower().replace('/', ' ').split()):
                        corrections[my_class] = match
                        break
            else:
                print('  No similar matches found')
            print()

    print(f'\n=== SUMMARY ===')
    total = len(my_classes)
    print(f'Found exact matches: {len(found)}/{total} ({len(found)/total*100:.1f}%)')
    print(f'Not found: {len(not_found)}')

    if corrections:
        print(f'\n=== SUGGESTED CORRECTIONS ===')
        for old_name, new_name in corrections.items():
            print(f'"{old_name}" → "{new_name}"')

    # Check for some specific architectural terms that might not be available
    architectural_terms = ['Balcony', 'Cornice', 'Pilaster', 'Parapet', 'Dormer']
    missing_architectural = [term for term in architectural_terms if term in not_found]
    if missing_architectural:
        print(f'\n=== NOTE ===')
        print('These architectural terms are not in the actual classes:')
        for term in missing_architectural:
            print(f'  - {term}')
        print('Consider using more general terms like "Building" or removing these specific architectural elements.')


if __name__ == '__main__':
    main()