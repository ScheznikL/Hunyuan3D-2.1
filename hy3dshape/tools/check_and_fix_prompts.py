#!/usr/bin/env python3
"""
Script to check if prompt files contain English text and optionally translate them.
Scans all prompt files in the preprocessed dataset and reports/fixes non-English prompts.
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
import unicodedata


def is_ascii(text):
    """Check if text contains only ASCII characters."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def contains_non_latin_script(text):
    """
    Detect if text contains non-Latin scripts (e.g., Kannada, Chinese, Arabic, etc.).
    Returns True if non-Latin characters are found.
    """
    if not text:
        return False
    
    # Count characters by script
    script_counts = defaultdict(int)
    
    for char in text:
        if char.isspace() or char in '.,;:!?()-[]{}"\'/\\':
            continue
        try:
            script_name = unicodedata.name(char).split()[0]
            script_counts[script_name] += 1
        except ValueError:
            continue
    
    # Check for common non-Latin scripts
    non_latin_scripts = {
        'KANNADA', 'DEVANAGARI', 'TAMIL', 'TELUGU', 'MALAYALAM', 'GUJARATI',
        'BENGALI', 'GURMUKHI', 'ORIYA', 'SINHALA',  # Indic scripts
        'ARABIC', 'HEBREW',  # Middle Eastern
        'CJK', 'HIRAGANA', 'KATAKANA', 'HANGUL',  # East Asian
        'CYRILLIC', 'GREEK',  # European non-Latin
        'THAI', 'LAO', 'MYANMAR', 'KHMER',  # Southeast Asian
    }
    
    for script, count in script_counts.items():
        if any(non_latin in script for non_latin in non_latin_scripts):
            return True
    
    return False


def count_non_ascii_chars(text):
    """Count how many non-ASCII characters are in the text."""
    return sum(1 for char in text if ord(char) > 127)


def get_text_stats(text):
    """Get statistics about the text for better analysis."""
    total_chars = len(text)
    non_ascii_count = count_non_ascii_chars(text)
    non_ascii_ratio = non_ascii_count / total_chars if total_chars > 0 else 0
    
    # Count actual alphabetic characters (not spaces, punctuation)
    alpha_chars = sum(1 for char in text if char.isalpha())
    non_ascii_alpha_ratio = non_ascii_count / alpha_chars if alpha_chars > 0 else 0
    
    return {
        'total_chars': total_chars,
        'non_ascii_count': non_ascii_count,
        'non_ascii_ratio': non_ascii_ratio,
        'alpha_chars': alpha_chars,
        'non_ascii_alpha_ratio': non_ascii_alpha_ratio
    }


def detect_language_issues(prompt_text):
    """
    Analyze prompt text and return issues found.
    Returns: (is_english, issues_list)
    
    Strategy:
    - A few non-ASCII chars (like em-dash \u2013, accents) are OK → English
    - Many non-Latin script characters → Non-English
    """
    issues = []
    
    if not prompt_text or not prompt_text.strip():
        return True, ["Empty prompt"]
    
    stats = get_text_stats(prompt_text)
    
    # Check for full non-Latin scripts (Kannada, Chinese, Arabic, etc.)
    has_non_latin = contains_non_latin_script(prompt_text)
    
    if has_non_latin:
        # This is clearly non-English (Kannada, Chinese, etc.)
        issues.append("Contains non-Latin script (e.g., Indic, Arabic, CJK)")
        return False, issues
    
    # Check for non-ASCII characters
    if stats['non_ascii_count'] > 0:
        # Threshold: if >20% of alphabetic characters are non-ASCII, likely non-English
        # Or if >10 non-ASCII chars total (arbitrary but reasonable)
        
        if stats['non_ascii_alpha_ratio'] > 0.20:
            issues.append(f"High proportion of non-ASCII characters ({stats['non_ascii_count']} chars, {stats['non_ascii_alpha_ratio']:.1%} of text)")
            return False, issues
        elif stats['non_ascii_count'] > 10:
            issues.append(f"Many non-ASCII characters ({stats['non_ascii_count']} chars)")
            return False, issues
        else:
            # Just a few non-ASCII chars (like em-dash, quotes, accents) - this is OK!
            # Don't report as an issue, just note it
            pass
    
    # Check for Unicode escape sequences in the actual text
    # But ignore common ones like \u2013 (em-dash), \u2019 (smart quote)
    if '\\u' in prompt_text:
        # This shouldn't happen in properly decoded text, but check anyway
        issues.append("Contains Unicode escape sequences (text may not be properly decoded)")
        return False, issues
    
    # If we got here, it's English (possibly with a few special characters)
    return True, issues


def scan_prompt_file(prompt_path):
    """
    Read and analyze a single prompt file.
    Returns: (is_english, prompt_data, issues)
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompt_text = data.get('prompt', '')
        is_english, issues = detect_language_issues(prompt_text)
        
        return is_english, data, issues
    
    except json.JSONDecodeError as e:
        return False, None, [f"JSON parse error: {str(e)}"]
    except Exception as e:
        return False, None, [f"Error reading file: {str(e)}"]


def scan_all_prompts(dataset_root, verbose=False):
    """
    Scan all prompt files in the dataset.
    Returns: (total_count, english_count, non_english_files)
    """
    prompt_pattern = Path(dataset_root) / "*" / "prompt" / "*_prompt"
    
    total_count = 0
    english_count = 0
    non_english_files = []
    
    print(f"Scanning prompts in: {dataset_root}")
    print("=" * 80)
    
    for prompt_file in Path(dataset_root).glob("*/prompt/*_prompt"):
        total_count += 1
        
        is_english, data, issues = scan_prompt_file(prompt_file)
        
        if is_english:
            english_count += 1
            if verbose:
                # Check if it has any non-ASCII chars but was still marked as English
                if data and 'prompt' in data:
                    non_ascii_count = count_non_ascii_chars(data['prompt'])
                    if non_ascii_count > 0:
                        print(f"✓ {prompt_file.name}: OK (English with {non_ascii_count} special chars like em-dash, quotes)")
                    else:
                        print(f"✓ {prompt_file.name}: OK")
                else:
                    print(f"✓ {prompt_file.name}: OK")
        else:
            non_english_files.append({
                'path': str(prompt_file),
                'filename': prompt_file.name,
                'issues': issues,
                'data': data
            })
            print(f"✗ {prompt_file.name}: {', '.join(issues)}")
            if data and 'prompt' in data and verbose:
                print(f"  Content: {data['prompt'][:100]}...")
    
    return total_count, english_count, non_english_files


def create_translation_plan(non_english_files, output_file):
    """
    Create a JSON file with prompts that need translation.
    This can be used with translation APIs or manual translation.
    """
    translation_plan = []
    
    for item in non_english_files:
        if item['data'] and 'prompt' in item['data']:
            translation_plan.append({
                'file': item['path'],
                'uid': item['data'].get('uid', 'unknown'),
                'original_prompt': item['data']['prompt'],
                'translated_prompt': '',  # To be filled in
                'issues': item['issues']
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translation_plan, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Translation plan saved to: {output_file}")
    print(f"  Contains {len(translation_plan)} items needing translation")


def apply_translations(translation_file, dry_run=False):
    """
    Apply translations from a filled-in translation plan file.
    """
    with open(translation_file, 'r', encoding='utf-8') as f:
        translation_plan = json.load(f)
    
    updated_count = 0
    skipped_count = 0
    
    for item in translation_plan:
        if not item.get('translated_prompt') or item['translated_prompt'].strip() == '':
            print(f"⊘ Skipping {item['uid']}: No translation provided")
            skipped_count += 1
            continue
        
        prompt_file = item['file']
        
        try:
            # Read original file
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update prompt
            old_prompt = data['prompt']
            data['prompt'] = item['translated_prompt']
            
            if dry_run:
                print(f"[DRY RUN] Would update {item['uid']}:")
                print(f"  FROM: {old_prompt[:80]}...")
                print(f"  TO:   {data['prompt'][:80]}...")
            else:
                # Write back
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"✓ Updated {item['uid']}")
            
            updated_count += 1
        
        except Exception as e:
            print(f"✗ Error updating {item['uid']}: {e}")
    
    print(f"\n{'[DRY RUN] Would update' if dry_run else 'Updated'}: {updated_count} files")
    print(f"Skipped (no translation): {skipped_count} files")


def main():
    parser = argparse.ArgumentParser(
        description='Check and fix non-English prompts in preprocessed dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan and report
  python check_and_fix_prompts.py --dataset /path/to/preprocessed
  
  # Create translation plan
  python check_and_fix_prompts.py --dataset /path/to/preprocessed --create-plan translations.json
  
  # Apply translations (dry run first)
  python check_and_fix_prompts.py --apply translations.json --dry-run
  
  # Apply translations for real
  python check_and_fix_prompts.py --apply translations.json
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to preprocessed dataset root (contains UUID folders with prompt subdirs)'
    )
    
    parser.add_argument(
        '--create-plan',
        type=str,
        metavar='OUTPUT_FILE',
        help='Create a JSON file with prompts needing translation'
    )
    
    parser.add_argument(
        '--apply',
        type=str,
        metavar='TRANSLATION_FILE',
        help='Apply translations from a filled-in translation plan JSON'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying files (use with --apply)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show details for all files, including those that pass'
    )
    
    args = parser.parse_args()
    
    # Mode 1: Apply translations
    if args.apply:
        print("Applying translations...")
        apply_translations(args.apply, dry_run=args.dry_run)
        return
    
    # Mode 2: Scan dataset
    if not args.dataset:
        parser.error("--dataset is required (unless using --apply)")
    
    total, english, non_english = scan_all_prompts(args.dataset, verbose=args.verbose)
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Total prompts:       {total}")
    print(f"  English prompts:     {english} ({100*english/total:.1f}%)")
    print(f"  Non-English prompts: {len(non_english)} ({100*len(non_english)/total:.1f}%)")
    
    if non_english and args.create_plan:
        create_translation_plan(non_english, args.create_plan)
        print(f"\nNext steps:")
        print(f"  1. Edit {args.create_plan} and fill in 'translated_prompt' fields")
        print(f"  2. Run: python {__file__} --apply {args.create_plan} --dry-run")
        print(f"  3. Run: python {__file__} --apply {args.create_plan}")


if __name__ == '__main__':
    main()