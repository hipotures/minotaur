# Documentation Update Process

This document describes the systematic approach for keeping documentation up-to-date with code changes by analyzing Git history and file modification times.

## <� Overview

The documentation update process involves:
1. **Analyzing file modification dates** vs Git commit history
2. **Identifying commits that affect documented $ARGUMENT** since last doc update
3. **Updating documentation** to reflect new functionality
4. **Updating metadata** (timestamps, commit references, change descriptions)

## =� Step-by-Step Process

### Step 1: Get Documentation File Modification Times

```bash
# Get modification dates for all documentation files
for file in docs/$ARGUMENT/*.md; do 
    echo "=== $file ==="; 
    stat -c "%y" "$file" 2>/dev/null || ls -la "$file" | awk '{print $6, $7, $8}'; 
done
```

### Step 2: Analyze Git History Since Documentation Updates

```bash
# Get the earliest modification time from docs
earliest_time=$(stat -c "%Y" docs/$ARGUMENT/*.md 2>/dev/null | sort -n | head -1)
if [ -n "$earliest_time" ]; then
    earliest_date=$(date -d "@$earliest_time" "+%Y-%m-%d %H:%M")
    echo "Analyzing commits since: $earliest_date"
    
    # Get commits since earliest doc modification
    git log --since="$earliest_date" --oneline --reverse
    
    # Get detailed commit times for analysis
    git log --since="$earliest_date" --pretty=format:"%h %ad %s" --date=format:"%H:%M:%S"
else
    echo "No existing documentation found, analyzing all recent commits"
    git log --oneline -20
fi
```

### Step 3: Check Current Documentation Coverage

For each key concept introduced in commits, check if it's documented:

```bash
# Check for specific concepts in documentation
echo "=== Checking documentation coverage ==="
for doc in docs/$ARGUMENT/*.md; do 
    if [ -f "$doc" ]; then
        echo "=== $(basename $doc) ==="; 
        
        # Count mentions of key terms based on component type
        case "$ARGUMENT" in
            "mcts")
                echo "UCB1 mentions:"; 
                grep -c "UCB1" "$doc" 2>/dev/null || echo "0"; 
                echo "Exploration mentions:"; 
                grep -c -i "exploration" "$doc" 2>/dev/null || echo "0"; 
                echo "Backpropagation mentions:"; 
                grep -c -i "backpropagation" "$doc" 2>/dev/null || echo "0"; 
                ;;
            "features")
                echo "Origin mentions:"; 
                grep -c "origin=" "$doc" 2>/dev/null || echo "0"; 
                echo "Auto-register mentions:"; 
                grep -c -i "auto.*register" "$doc" 2>/dev/null || echo "0"; 
                echo "Train features mentions:"; 
                grep -c -i "train.*feature" "$doc" 2>/dev/null || echo "0"; 
                ;;
            "db")
                echo "Repository mentions:"; 
                grep -c -i "repository" "$doc" 2>/dev/null || echo "0"; 
                echo "Migration mentions:"; 
                grep -c -i "migration" "$doc" 2>/dev/null || echo "0"; 
                echo "DuckDB mentions:"; 
                grep -c "DuckDB" "$doc" 2>/dev/null || echo "0"; 
                ;;
            *)
                echo "Component mentions:"; 
                grep -c -i "$ARGUMENT" "$doc" 2>/dev/null || echo "0"; 
                ;;
        esac
        echo; 
    fi
done
```

### Step 4: Update Documentation Content

For each file that needs updates, add sections covering:

#### Key Patterns to Add (Examples for different components):

**For Features ($ARGUMENT = features):**
- Origin Field System
- Auto-Registration Process
- API Documentation for Developers

**For MCTS ($ARGUMENT = mcts):**
- Algorithm updates (UCB1, tree traversal, node expansion)
- Configuration changes in YAML files
- Performance optimizations

**For Database ($ARGUMENT = db):**
- Schema changes and migrations
- Repository pattern updates
- Connection pooling improvements

### Step 5: Update Documentation Metadata

Update the header metadata in each file:

```bash
# Update timestamps and commit references for all updated files
current_time=$(date "+%Y-%m-%d %H:%M")
latest_commit=$(git rev-parse --short HEAD)

# Handle different filename patterns based on component
case "$ARGUMENT" in
    "features")
        pattern="FEATURES_"
        ;;
    "mcts")
        pattern="MCTS_"
        ;;
    "db")
        pattern="DB_"
        ;;
    *)
        pattern=""
        ;;
esac

for file in docs/$ARGUMENT/${pattern}*.md; do 
    if [ -f "$file" ]; then
        sed -i "s/Last Updated: [0-9-]* [0-9:]*/Last Updated: $current_time/" "$file"
        sed -i "s/Compatible with commit: [a-f0-9]*/Compatible with commit: $latest_commit/" "$file"
    fi
done
```

**Update change descriptions** to reflect what was added:
```markdown
<!-- 
Documentation Status: CURRENT
Last Updated: YYYY-MM-DD HH:MM
Compatible with commit: XXXXXXX
Changes: [Description of what was added/updated]
-->
```

### Step 6: Commit Documentation Updates

```bash
# Add documentation files (force if needed due to .gitignore)
git add -f docs/$ARGUMENT/

# Commit with detailed message
git commit -m "docs: update $ARGUMENT documentation with latest system changes

Updated all $ARGUMENT documentation to reflect recent code changes:

## Updated Documents:
$(ls -1 docs/$ARGUMENT/*.md 2>/dev/null | xargs -I {} basename {} | sed 's/^/- /')

## Key Updates:
- Synchronized with latest code changes up to commit $(git rev-parse --short HEAD)
- Updated API documentation and examples
- Added new features and configuration options
- Improved clarity and completeness

## Coverage:
All commits affecting $ARGUMENT components since last documentation update
are now reflected in the documentation.

Documentation is current with commit $(git rev-parse --short HEAD).

> Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

## = Analysis Criteria

### When to Update Documentation

Update documentation when commits introduce:

1. **New APIs or Parameters** (e.g., new function parameters, configuration options)
2. **New System Components** (e.g., new modules, classes, or features)
3. **New Processes** (e.g., new workflows or procedures)
4. **Breaking Changes** (e.g., API changes, renamed functions)
5. **Major Feature Additions** (e.g., new capabilities)

### What to Document

For each new feature, ensure documentation covers:

- **Conceptual Overview**: What it is and why it exists
- **API Documentation**: Parameters, usage examples, return values
- **Integration Points**: How it connects with existing systems
- **Developer Guidelines**: Best practices, gotchas, patterns
- **Examples**: Real code snippets and expected outputs

## =� Tools and Commands Reference

### File Analysis Commands
```bash
# Get file modification times
stat -c "%y" docs/$ARGUMENT/*.md

# Check file content for keywords
grep -n -i "keyword" docs/$ARGUMENT/*.md

# Count occurrences of concepts
grep -c "pattern" docs/$ARGUMENT/*.md
```

### Git Analysis Commands
```bash
# Get commits since date
git log --since="YYYY-MM-DD HH:MM" --oneline

# Get detailed commit info
git log --pretty=format:"%h %ad %s" --date=format:"%H:%M:%S"

# Find commits affecting specific files based on component
case "$ARGUMENT" in
    "mcts")
        search_paths="src/mcts*.py src/tree*.py src/node*.py config/mcts*.yaml"
        ;;
    "features")
        search_paths="src/features/ src/feature_*.py"
        ;;
    "db")
        search_paths="src/db/ src/repositories/ migrations/"
        ;;
    *)
        search_paths="src/"
        ;;
esac
git log --oneline -- $search_paths

# Get current commit hash
git rev-parse --short HEAD
```

### Batch Update Commands
```bash
# Update timestamps in all docs
current_time=$(date "+%Y-%m-%d %H:%M")
sed -i "s/Last Updated: .*/Last Updated: $current_time/" docs/$ARGUMENT/*.md

# Update commit references
latest_commit=$(git rev-parse --short HEAD)
sed -i "s/Compatible with commit: .*/Compatible with commit: $latest_commit/" docs/$ARGUMENT/*.md
```

## =� Example: Complete Update Session

```bash
# 1. Analyze current state
echo "=== Documentation Files ==="
ls -la docs/$ARGUMENT/*.md 2>/dev/null | awk '{print $6, $7, $8, $9}'

echo "=== Recent Commits ==="
# Get component-specific search paths
case "$ARGUMENT" in
    "mcts")
        search_paths="src/mcts*.py src/tree*.py src/node*.py config/mcts*.yaml"
        ;;
    "features")
        search_paths="src/features/ src/feature_*.py"
        ;;
    "db")
        search_paths="src/db/ src/repositories/ migrations/"
        ;;
    *)
        search_paths="src/"
        ;;
esac
git log --oneline -10 -- $search_paths

# 2. Check documentation coverage
echo "=== Checking for new concepts ==="
# Define concepts based on component type
case "$ARGUMENT" in
    "mcts")
        concepts=("UCB1" "exploration" "backpropagation")
        ;;
    "features")
        concepts=("origin=" "auto.*register" "train.*feature")
        ;;
    "db")
        concepts=("repository" "migration" "DuckDB")
        ;;
    *)
        concepts=("$ARGUMENT")
        ;;
esac

for concept in "${concepts[@]}"; do
    echo "--- $concept ---"
    grep -l "$concept" docs/$ARGUMENT/*.md 2>/dev/null || echo "Not found in docs"
done

# 3. Update files (manual editing with identified gaps)
# ... edit files to add missing content ...

# 4. Update metadata
current_time=$(date "+%Y-%m-%d %H:%M")
latest_commit=$(git rev-parse --short HEAD)
for file in docs/$ARGUMENT/*.md; do
    if [ -f "$file" ]; then
        sed -i "s/Last Updated: .*/Last Updated: $current_time/" "$file"
        sed -i "s/Compatible with commit: .*/Compatible with commit: $latest_commit/" "$file"
    fi
done

# 5. Commit changes
git add -f docs/$ARGUMENT/
git commit -m "docs: update $ARGUMENT documentation with latest system changes"
```

## = Regular Maintenance

**Recommended Schedule:**
- **After major feature commits**: Immediate documentation update
- **Weekly**: Review and update any missed changes
- **Before releases**: Comprehensive documentation review

**Quick Check Command:**
```bash
# Check if docs are current with latest commits
# Define search paths based on component
case "$ARGUMENT" in
    "mcts")
        search_paths="src/mcts*.py src/tree*.py src/node*.py config/mcts*.yaml"
        ;;
    "features")
        search_paths="src/features/ src/feature_*.py"
        ;;
    "db")
        search_paths="src/db/ src/repositories/ migrations/"
        ;;
    *)
        search_paths="src/"
        ;;
esac

latest_doc_time=$(stat -c "%Y" docs/$ARGUMENT/*.md 2>/dev/null | sort -n | tail -1)
latest_commit_time=$(git log -1 --format="%ct" -- $search_paths)

if [ -n "$latest_doc_time" ] && [ -n "$latest_commit_time" ]; then
    if [ $latest_commit_time -gt $latest_doc_time ]; then
        echo "Documentation may be outdated - review needed"
    else
        echo "Documentation appears current"
    fi
else
    echo "Unable to determine documentation status"
fi
```

This process ensures documentation stays synchronized with code changes and provides comprehensive coverage of new features and API changes.

## 🚀 Usage

This command will analyze Git history and guide you through updating the `$ARGUMENT` documentation:

```
/project:update-docs-from-git.md $ARGUMENT
```

Where `$ARGUMENT` can be:
- `mcts` - for MCTS algorithm documentation
- `features` - for feature engineering documentation  
- `db` - for database architecture documentation
- Any other subdirectory under `docs/`