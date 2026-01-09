# Contributing to RL Crop Segmentation

Thank you for your interest in contributing to the RL-based Crop Segmentation project! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU/CPU)
- Relevant logs or error messages

### Suggesting Features

Feature requests are welcome! Please:
- Check existing issues first
- Clearly describe the feature and its use case
- Explain why it would be beneficial
- Provide examples if possible

### Code Contributions

1. **Fork the repository**

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation
   - Add comments where necessary

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   black src/ utils/
   flake8 src/ utils/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

   Use conventional commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Refactor:` for code refactoring
   - `Docs:` for documentation changes

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a PR on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/examples if applicable

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rl_segmentation.git
cd rl_segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for all functions/classes
- Format code with Black

Example:

```python
def process_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process satellite image.

    Args:
        image: Input image array (C, H, W)
        normalize: Whether to normalize to [0, 1]

    Returns:
        Processed image array

    Raises:
        ValueError: If image has invalid shape
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")

    if normalize:
        image = image.astype(np.float32) / 255.0

    return image
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

```python
# tests/test_data_loader.py
import pytest
from src.data_loader import GeoDataLoader

def test_load_image():
    loader = GeoDataLoader(
        image_path="test_data/image.tif",
        shapefile_path="test_data/mask.shp"
    )
    image = loader.load_image()
    assert image is not None
    assert len(image.shape) == 3
```

## Documentation

- Update README.md if adding new features
- Add docstrings to all new functions/classes
- Update config files if adding new parameters
- Add examples to notebooks if applicable

## Areas for Contribution

We especially welcome contributions in:

### High Priority
- [ ] Additional RL algorithms (A3C, SAC, TD3)
- [ ] Multi-class segmentation support
- [ ] Data augmentation strategies
- [ ] Uncertainty estimation
- [ ] Model compression/quantization

### Medium Priority
- [ ] Additional reward functions
- [ ] Curriculum learning strategies
- [ ] Transfer learning support
- [ ] Multi-GPU training
- [ ] Distributed training

### Nice to Have
- [ ] Web interface for visualization
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Additional datasets/benchmarks
- [ ] Performance optimizations

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Reach out to maintainers

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to learn and improve!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🚀
