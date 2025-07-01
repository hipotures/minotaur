# Update $ARGUMENT Documentation from Git History

## 🎯 Objective

Analyze Git history since the last documentation update for `docs/$ARGUMENT/` and update all documentation files to reflect code changes.

## 📋 Step-by-Step Process

### Step 1: Get Documentation File Modification Times

```bash
# Get modification dates for all $ARGUMENT documentation files
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

### Step 3: Identify Commits Affecting $ARGUMENT Components

```bash
# Find commits that affect $ARGUMENT-related code
echo "=== Commits affecting $ARGUMENT components ==="

# Define search paths based on the documentation type
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

git log --oneline -- $search_paths | head -20
```

### Step 4: Check Current Documentation Coverage

```bash
# Check if key concepts are documented
echo "=== Checking documentation coverage ==="
for doc in docs/$ARGUMENT/*.md; do 
    if [ -f "$doc" ]; then
        echo "--- $(basename $doc) ---"
        # Count mentions of key terms based on component type
        case "$ARGUMENT" in
            "mcts")
                echo "Key MCTS concepts:"
                echo -n "  UCB1 mentions: "; grep -c "UCB1" "$doc" 2>/dev/null || echo "0"
                echo -n "  Exploration mentions: "; grep -c -i "exploration" "$doc" 2>/dev/null || echo "0"
                echo -n "  Backpropagation mentions: "; grep -c -i "backpropagation" "$doc" 2>/dev/null || echo "0"
                ;;
            "features")
                echo "Key Features concepts:"
                echo -n "  Origin field mentions: "; grep -c "origin=" "$doc" 2>/dev/null || echo "0"
                echo -n "  Auto-register mentions: "; grep -c -i "auto.*register" "$doc" 2>/dev/null || echo "0"
                echo -n "  Feature catalog mentions: "; grep -c -i "feature.*catalog" "$doc" 2>/dev/null || echo "0"
                ;;
            "db")
                echo "Key Database concepts:"
                echo -n "  Repository mentions: "; grep -c -i "repository" "$doc" 2>/dev/null || echo "0"
                echo -n "  Migration mentions: "; grep -c -i "migration" "$doc" 2>/dev/null || echo "0"
                echo -n "  DuckDB mentions: "; grep -c "DuckDB" "$doc" 2>/dev/null || echo "0"
                ;;
        esac
        echo
    fi
done
```

### Step 5: Update Documentation Content

Based on the analysis above, update the documentation files in `docs/$ARGUMENT/` to include:

1. **New functionality** introduced in recent commits
2. **API changes** or new parameters
3. **Configuration updates** 
4. **Integration changes** with other components
5. **Performance improvements** or optimizations

#### Component-Specific Update Guidelines:

##### For MCTS Documentation ($ARGUMENT = mcts)
- Algorithm updates (UCB1, tree traversal, node expansion)
- Configuration changes in YAML files
- Performance optimizations and memory management
- Session management and resume functionality

##### For Features Documentation ($ARGUMENT = features)
- New feature operations and categories
- Origin field system updates
- Auto-registration mechanisms
- Domain-specific feature implementations

##### For Database Documentation ($ARGUMENT = db)
- Schema changes and migrations
- Repository pattern updates
- Connection pooling improvements
- Query optimizations and indexes

### Step 6: Update Documentation Metadata

```bash
# Update timestamps and commit references for all updated files
current_time=$(date "+%Y-%m-%d %H:%M")
latest_commit=$(git rev-parse --short HEAD)

for file in docs/$ARGUMENT/*.md; do 
    if [ -f "$file" ]; then
        # Update timestamp
        sed -i "s/Last Updated: [0-9-]* [0-9:]*/Last Updated: $current_time/" "$file"
        
        # Update commit reference
        sed -i "s/Compatible with commit: [a-f0-9]*/Compatible with commit: $latest_commit/" "$file"
        
        echo "Updated metadata in: $file"
    fi
done
```

### Step 7: Generate Commit Message

```bash
# Create temporary file for commit message
cat > /tmp/commit_msg.txt << EOF
docs: update $ARGUMENT documentation with latest system changes

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

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF

echo "Commit message prepared in /tmp/commit_msg.txt"
```

### Step 8: Commit Documentation Updates

```bash
# Add documentation files (force if needed due to .gitignore)
git add -f docs/$ARGUMENT/

# Review changes
echo "=== Changes to be committed ==="
git status
git diff --cached --stat

# Commit with the generated message
git commit -F /tmp/commit_msg.txt
```

## 🔍 Quick Validation

```bash
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

# Verify documentation is up to date
latest_doc_time=$(stat -c "%Y" docs/$ARGUMENT/*.md 2>/dev/null | sort -n | tail -1)
latest_code_time=$(git log -1 --format="%ct" -- $search_paths)

if [ -n "$latest_doc_time" ] && [ -n "$latest_code_time" ]; then
    if [ $latest_code_time -gt $latest_doc_time ]; then
        echo "⚠️  Documentation may still be outdated - review needed"
    else
        echo "✅ Documentation appears current"
    fi
else
    echo "❓ Unable to determine documentation status"
fi
```

## 🛠️ Tools and Commands Reference

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

# Find commits affecting specific files
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

## 📝 Example: Complete Update Session

```bash
# Set the component to update
ARGUMENT="$ARGUMENT"  # Will be replaced by actual argument

# 1. Analyze current state
echo "=== Documentation Files ==="
ls -la docs/$ARGUMENT/*.md 2>/dev/null | awk '{print $6, $7, $8, $9}'

echo -e "\n=== Recent Commits ==="
case "$ARGUMENT" in
    "mcts") search_paths="src/mcts*.py src/tree*.py src/node*.py config/mcts*.yaml" ;;
    "features") search_paths="src/features/ src/feature_*.py" ;;
    "db") search_paths="src/db/ src/repositories/ migrations/" ;;
    *) search_paths="src/" ;;
esac
git log --oneline -10 -- $search_paths

# 2. Check documentation coverage
echo -e "\n=== Checking documentation coverage ==="
for doc in docs/$ARGUMENT/*.md; do
    if [ -f "$doc" ]; then
        echo "$(basename $doc) exists"
    fi
done

# 3. Update files (manual editing based on analysis)
echo -e "\n=== Update documentation files manually based on the analysis above ==="

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

## 🔄 Regular Maintenance

**Recommended Schedule:**
- **After major commits**: Immediate documentation update
- **Weekly**: Review and update any missed changes
- **Before releases**: Comprehensive documentation review

**Quick Check Command:**
```bash
# Check if docs are current with latest commits
ARGUMENT="$ARGUMENT"  # Will be replaced by actual argument
case "$ARGUMENT" in
    "mcts") search_paths="src/mcts*.py src/tree*.py src/node*.py config/mcts*.yaml" ;;
    "features") search_paths="src/features/ src/feature_*.py" ;;
    "db") search_paths="src/db/ src/repositories/ migrations/" ;;
    *) search_paths="src/" ;;
esac

latest_doc_time=$(stat -c "%Y" docs/$ARGUMENT/*.md 2>/dev/null | sort -n | tail -1)
latest_commit_time=$(git log -1 --format="%ct" -- $search_paths)

if [ -n "$latest_doc_time" ] && [ -n "$latest_commit_time" ]; then
    if [ $latest_commit_time -gt $latest_doc_time ]; then
        echo "Documentation may be outdated - review needed"
    else
        echo "Documentation appears current"
    fi
fi
```

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