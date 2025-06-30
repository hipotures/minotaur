# Documentation Update Process

This document describes the systematic approach for keeping documentation up-to-date with code changes by analyzing Git history and file modification times.

## 🎯 Overview

The documentation update process involves:
1. **Analyzing file modification dates** vs Git commit history
2. **Identifying commits that affect documented features** since last doc update
3. **Updating documentation** to reflect new functionality
4. **Updating metadata** (timestamps, commit references, change descriptions)

## 📋 Step-by-Step Process

### Step 1: Get Documentation File Modification Times

```bash
# Get modification dates for all documentation files
for file in docs/features/*.md; do 
    echo "=== $file ==="; 
    stat -c "%y" "$file" 2>/dev/null || ls -la "$file" | awk '{print $6, $7, $8}'; 
done

# Example output:
# === docs/features/FEATURES_OVERVIEW.md ===
# 2025-06-30 21:29:31.493570952 +0200
```

### Step 2: Analyze Git History Since Documentation Updates

```bash
# Get commits since earliest doc modification (adjust date as needed)
git log --since="2025-06-30 14:00" --oneline --reverse

# Get detailed commit times for analysis
git log --since="2025-06-30 14:00" --pretty=format:"%h %ad %s" --date=format:"%H:%M:%S"

# Example output:
# c007d87 21:30:05 feat: add origin field to feature catalog for complete feature management
# a14ffe7 23:21:39 feat: auto-register train features during dataset import
# bcc1217 23:28:38 fix: update Titanic custom features to use lowercase column names
```

### Step 3: Check Current Documentation Coverage

For each key concept introduced in commits, check if it's documented:

```bash
# Check for specific concepts in documentation
for doc in docs/features/*.md; do 
    echo "=== $(basename $doc) ==="; 
    echo "Origin mentions:"; 
    grep -c "origin=" "$doc" 2>/dev/null || echo "0"; 
    echo "Auto-register mentions:"; 
    grep -c -i "auto.*register" "$doc" 2>/dev/null || echo "0"; 
    echo "Train features mentions:"; 
    grep -c -i "train.*feature" "$doc" 2>/dev/null || echo "0"; 
    echo; 
done
```

### Step 4: Update Documentation Content

For each file that needs updates, add sections covering:

#### Key Patterns to Add:

**Origin Field System:**
```markdown
## Feature Classification System

All features in the Minotaur system are automatically classified by **origin** during generation:

### Origin Types
- **`origin='train'`**: Original dataset columns (automatically registered during dataset import)
- **`origin='generic'`**: Domain-agnostic operations (statistical, polynomial, binning, ranking, etc.)
- **`origin='custom'`**: Domain-specific operations (competition/problem-specific features)
```

**Auto-Registration Process:**
```markdown
### Auto-Registration Process

**During Dataset Registration**:
1. **Dataset Validation**: Schema analysis and data type detection
2. **Train Features Registration**: Original dataset columns automatically registered with `origin='train'`
3. **Feature Generation**: All operations run on 100% of data
   - **Generic Features**: Domain-agnostic operations with `origin='generic'`
   - **Custom Features**: Domain-specific operations with `origin='custom'`
4. **Signal Detection**: No-signal features filtered automatically
5. **DuckDB Storage**: Features stored in separate tables
6. **Feature Catalog**: All features registered with origin classification and metadata
```

**API Documentation for Developers:**
```markdown
### Auto-Registration API

**All feature operations inherit auto-registration functionality**:

```python
def generate_features(self, df: pd.DataFrame, 
                     auto_register: bool = True, 
                     origin: str = 'generic',
                     **kwargs) -> Dict[str, pd.Series]:
    """
    Generate features with automatic catalog registration.
    
    Args:
        df: Input dataframe
        auto_register: Whether to register features in catalog
        origin: Feature origin classification ('generic', 'custom', 'train')
        **kwargs: Operation-specific parameters
    """
```

### Step 5: Update Documentation Metadata

Update the header metadata in each file:

```bash
# Update timestamps and commit references for all updated files
current_time=$(date "+%Y-%m-%d %H:%M")
latest_commit=$(git rev-parse --short HEAD)

for file in docs/features/FEATURES_*.md; do 
    sed -i "s/Last Updated: [0-9-]* [0-9:]*/Last Updated: $current_time/" "$file"
    sed -i "s/Compatible with commit: [a-f0-9]*/Compatible with commit: $latest_commit/" "$file"
done
```

**Update change descriptions** to reflect what was added:
```markdown
<!-- 
Documentation Status: CURRENT
Last Updated: 2025-06-30 23:40
Compatible with commit: bcc1217
Changes: Added origin field system, auto-registration mechanism, and dynamic feature categories
-->
```

### Step 6: Commit Documentation Updates

```bash
# Add documentation files (force if needed due to .gitignore)
git add -f docs/features/

# Commit with detailed message
git commit -m "docs: update feature engineering documentation with latest system changes

Updated all feature documentation to reflect major system enhancements:

## Updated Documents:
- FEATURES_INTEGRATION.md: Added origin field system and auto-registration flow
- FEATURES_OPERATIONS.md: Added feature classification system and auto-registration API
- FEATURES_DEVELOPMENT.md: Added developer guide for origin field and auto-registration
- FEATURES_OVERVIEW.md: Enhanced auto-registration process documentation

## Key Additions:
- **Origin Field System**: Complete documentation of train/generic/custom classification
- **Auto-Registration Mechanism**: Automatic feature catalog management during dataset import
- **Dynamic Feature Categories**: Database-driven category mapping system
- **API Documentation**: Developer guidelines for new origin and auto-registration parameters

## Coverage:
All major commits since doc creation are now documented:
- c007d87: Origin field implementation
- a14ffe7: Auto-registration of train features  
- d5d6f99: Dynamic feature category system
- bcc1217: Custom features case sensitivity fix

Documentation is now current with commit bcc1217 and reflects the complete
feature engineering system with 414 total features (10 train + 47 custom + 357 generic).

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## 🔍 Analysis Criteria

### When to Update Documentation

Update documentation when commits introduce:

1. **New APIs or Parameters** (e.g., `auto_register`, `origin` parameters)
2. **New System Components** (e.g., origin field, feature catalog)
3. **New Processes** (e.g., auto-registration during dataset import)
4. **Breaking Changes** (e.g., column name case changes)
5. **Major Feature Additions** (e.g., dynamic category system)

### What to Document

For each new feature, ensure documentation covers:

- **Conceptual Overview**: What it is and why it exists
- **API Documentation**: Parameters, usage examples, return values
- **Integration Points**: How it connects with existing systems
- **Developer Guidelines**: Best practices, gotchas, patterns
- **Examples**: Real code snippets and expected outputs

## 🛠️ Tools and Commands Reference

### File Analysis Commands
```bash
# Get file modification times
stat -c "%y" docs/features/*.md

# Check file content for keywords
grep -n -i "keyword" docs/features/*.md

# Count occurrences of concepts
grep -c "origin=" docs/features/*.md
```

### Git Analysis Commands
```bash
# Get commits since date
git log --since="YYYY-MM-DD HH:MM" --oneline

# Get detailed commit info
git log --pretty=format:"%h %ad %s" --date=format:"%H:%M:%S"

# Find commits affecting specific files
git log --oneline -- src/features/

# Get current commit hash
git rev-parse --short HEAD
```

### Batch Update Commands
```bash
# Update timestamps in all docs
current_time=$(date "+%Y-%m-%d %H:%M")
sed -i "s/Last Updated: .*/Last Updated: $current_time/" docs/features/*.md

# Update commit references
latest_commit=$(git rev-parse --short HEAD)
sed -i "s/Compatible with commit: .*/Compatible with commit: $latest_commit/" docs/features/*.md
```

## 📝 Example: Complete Update Session

```bash
# 1. Analyze current state
echo "=== Documentation Files ==="
ls -la docs/features/*.md | awk '{print $6, $7, $8, $9}'

echo "=== Recent Commits ==="
git log --since="2025-06-30 14:00" --oneline

# 2. Check documentation coverage
echo "=== Checking for new concepts ==="
for concept in "origin=" "auto.*register" "train.*feature"; do
    echo "--- $concept ---"
    grep -l "$concept" docs/features/*.md || echo "Not found in docs"
done

# 3. Update files (manual editing with identified gaps)
# ... edit files to add missing content ...

# 4. Update metadata
current_time=$(date "+%Y-%m-%d %H:%M")
latest_commit=$(git rev-parse --short HEAD)
for file in docs/features/*.md; do
    sed -i "s/Last Updated: .*/Last Updated: $current_time/" "$file"
    sed -i "s/Compatible with commit: .*/Compatible with commit: $latest_commit/" "$file"
done

# 5. Commit changes
git add -f docs/features/
git commit -m "docs: update documentation with latest system changes"
```

## 🔄 Regular Maintenance

**Recommended Schedule:**
- **After major feature commits**: Immediate documentation update
- **Weekly**: Review and update any missed changes
- **Before releases**: Comprehensive documentation review

**Quick Check Command:**
```bash
# Check if docs are current with latest commits
latest_doc_time=$(stat -c "%Y" docs/features/*.md | sort -n | tail -1)
latest_commit_time=$(git log -1 --format="%ct")
if [ $latest_commit_time -gt $latest_doc_time ]; then
    echo "Documentation may be outdated - review needed"
else
    echo "Documentation appears current"
fi
```

This process ensures documentation stays synchronized with code changes and provides comprehensive coverage of new features and API changes.