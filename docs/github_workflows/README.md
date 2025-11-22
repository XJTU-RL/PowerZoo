# GitHub Workflows

This directory contains GitHub Actions workflow configurations that should be
placed in `.github/workflows/` directory.

## Installation

To enable CI/CD, copy these files to the `.github/` directory:

```bash
# From project root
mkdir -p .github/workflows
cp docs/github_workflows/ci.yml .github/workflows/
cp docs/github_workflows/pr-check.yml .github/workflows/
cp docs/github_workflows/dependabot.yml .github/
```

## Workflow Descriptions

### ci.yml
Main CI pipeline that runs on push and pull requests:
- Code quality checks with Ruff
- Multi-Python version testing (3.9, 3.10, 3.11)
- Stackelberg environment integration tests
- Optional type checking with mypy

### pr-check.yml
Pull request validation:
- Large file detection
- YAML config validation
- Python syntax checks
- Module structure verification

### dependabot.yml
Automated dependency updates:
- Weekly Python dependency updates
- GitHub Actions version updates

## Permissions Note

These files require `workflows` write permission to be added via git push.
If you're a repository admin, you can either:
1. Push these files manually
2. Add via the GitHub web interface
