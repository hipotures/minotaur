# TODO - Minotaur Project

## UX Improvements

### Bash Autocompletion for manager.py
- [ ] Create bash completion script for `./manager.py` command
- [ ] Support autocompletion for:
  - Modules: `datasets`, `features`, `sessions`, `analytics`, etc.
  - Module commands: `--list`, `--register`, `--details`, etc.
  - Parameters: dataset names, file paths, etc.
- [ ] Install completion script to `/etc/bash_completion.d/` or user's `.bashrc`
- [ ] Test with: `./manager.py datasets <TAB><TAB>` should show available options

## Future Enhancements

### Performance
- [ ] Optimize feature generation for large datasets (>1M records)
- [ ] Add progress bars for long operations
- [ ] Implement background processing for dataset registration

### Database Refactoring
- [ ] Refactor SQL to use Views for complex queries
- [ ] Implement VIRTUAL columns for computed values (e.g., derived metrics)
- [ ] Add STORED columns for frequently accessed calculations
- [ ] Create materialized views for performance-critical aggregations
- [ ] Optimize session_resume_params view with proper indexing
- [ ] Consider using generated/computed columns for total_iterations to avoid off-by-one errors

### Features
- [ ] Add dataset comparison functionality
- [ ] Implement dataset versioning system
- [ ] Add data quality metrics and validation

### Developer Experience
- [ ] Add comprehensive API documentation
- [ ] Create interactive CLI help system
- [ ] Add configuration validation tools