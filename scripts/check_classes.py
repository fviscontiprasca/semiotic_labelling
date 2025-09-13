#!/usr/bin/env python3
"""
Check `my_classes.txt` against Open Images V7 class descriptions.

This script reads the class list from `my_classes.txt` (repo root) and compares each
entry against `data/oid_urban/class-descriptions-boxable.csv`. It prints exact
matches and suggests similar matches when exact ones are missing.
"""

import sys
from pathlib import Path
import pandas as pd


def find_similar_matches(df, term, max_results=5):
    """Return up to max_results similar display names for term."""
    keywords = [w.lower() for w in term.replace('/', ' ').replace('(', '').replace(')', '').split() if w.strip()]
    matches = []
    for kw in keywords:
        if len(kw) <= 2:
            continue
        found = df[df['DisplayName'].str.contains(kw, case=False, na=False)]['DisplayName'].tolist()
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
    class_desc_file = repo_root / 'data' / 'oid_urban' / 'extended_class-descriptions.csv'

    if not classes_file.exists():
        print(f'Error: {classes_file} not found', file=sys.stderr)
        return
    if not class_desc_file.exists():
        print(f'Error: {class_desc_file} not found', file=sys.stderr)
        return

    with open(classes_file, 'r', encoding='utf-8') as f:
        my_classes = [line.strip() for line in f if line.strip()]

    # Read extended CSV robustly (handles quoted/comma-containing names)
    # Read the CSV trying common encodings (some extended files contain non-UTF8 bytes)
    encodings_to_try = ['utf-8', 'cp1252', 'latin-1']
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(class_desc_file, header=None, names=['LabelName', 'DisplayName'], dtype=str, engine='python', encoding=enc)
            # If read succeeds, break
            break
        except Exception:
            df = None
            continue
    if df is None:
        print(f'Error: failed to read {class_desc_file} with tried encodings {encodings_to_try}', file=sys.stderr)
        return

    # Normalize and clean display names (remove weird control characters)
    df['DisplayName'] = df['DisplayName'].astype(str).str.strip().str.replace(r'[\x00-\x1F\x7F]', '', regex=True)

    print('=== CHECKING YOUR CLASSES AGAINST OID V7 ===\n')

    found = []
    not_found = []
    corrections = {}

    for my_class in my_classes:
        exact_match = df[df['DisplayName'] == my_class]
        if not exact_match.empty:
            found.append(my_class)
            print(f'✓ FOUND: {my_class}')
        else:
            not_found.append(my_class)
            print(f'✗ NOT FOUND: {my_class}')
            similar = find_similar_matches(df, my_class, max_results=5)
            if similar:
                print('  Possible matches:')
                for match in similar:
                    label_id = df[df['DisplayName'] == match].iloc[0]['LabelName']
                    print(f'    - {match} ({label_id})')
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

    # Check for some specific architectural terms that might not be in OID
    architectural_terms = ['Balcony', 'Cornice', 'Pilaster', 'Parapet', 'Dormer']
    missing_architectural = [term for term in architectural_terms if term in not_found]
    if missing_architectural:
        print(f'\n=== NOTE ===')
        print('These architectural terms are not in Open Images V7:')
        for term in missing_architectural:
            print(f'  - {term}')
        print('Consider using more general terms like "Building" or removing these specific architectural elements.')


if __name__ == '__main__':
    main()