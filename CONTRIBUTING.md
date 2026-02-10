# Contributing to Fake News Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Screenshots if applicable

### Suggesting Enhancements

Feature requests are welcome! Please create an issue with:
- Clear description of the enhancement
- Use cases and benefits
- Implementation ideas (if any)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow Python PEP 8 style guide
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python -m pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/YourFeatureName
   ```

7. **Create a Pull Request**
   - Provide clear description
   - Reference related issues
   - Wait for review

## Development Setup

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
# Clone your fork
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## Code Style

### Python
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Maximum line length: 100 characters

### Documentation
- Update README.md for new features
- Add docstrings with parameter descriptions
- Include examples where helpful

### Commit Messages
Use clear, descriptive commit messages:
- `Add: [feature description]`
- `Fix: [bug description]`
- `Update: [what was updated]`
- `Docs: [documentation changes]`

## Testing

Before submitting a PR:
- Ensure all existing tests pass
- Add tests for new features
- Test on multiple Python versions if possible

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing! 🎉
