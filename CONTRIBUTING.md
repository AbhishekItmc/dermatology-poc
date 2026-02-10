# Contributing Guide

Thank you for contributing to the Dermatological Analysis PoC project!

## Development Workflow

### 1. Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Emergency fixes for production

### 2. Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### 3. Making Changes

1. Write code following the project style guide
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass locally

### 4. Running Tests Locally

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm test

# Or use Make commands
make test-backend
make test-frontend
```

### 5. Code Quality Checks

#### Backend (Python)

```bash
# Linting
flake8 app --count --select=E9,F63,F7,F82 --show-source --statistics

# Formatting
black app

# Type checking
mypy app
```

#### Frontend (TypeScript)

```bash
# Linting
npm run lint

# Formatting
npm run format
```

### 6. Committing Changes

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```bash
git commit -m "feat(detection): implement pigmentation detection model

- Add U-Net architecture with attention
- Implement severity classification
- Add unit tests for detection pipeline

Validates: Requirements 1.1, 1.2"
```

### 7. Pushing Changes

```bash
git push origin feature/your-feature-name
```

### 8. Creating a Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your feature branch
4. Fill in the PR template:
   - Description of changes
   - Related requirements/tasks
   - Testing performed
   - Screenshots (if UI changes)
5. Request review from team members

### 9. Code Review Process

- At least one approval required
- All CI checks must pass
- No merge conflicts
- Code coverage must not decrease

### 10. Merging

Once approved:
```bash
# Squash and merge for feature branches
# Merge commit for release branches
```

## Code Style Guidelines

### Python (Backend)

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes
- Format with Black

Example:
```python
def detect_pigmentation(image: np.ndarray) -> List[PigmentationArea]:
    """
    Detect pigmentation areas in a facial image.
    
    Args:
        image: Normalized facial image tensor (H x W x 3)
    
    Returns:
        List of detected pigmentation areas with metrics
    
    Requirements: 1.1, 1.2
    """
    # Implementation
    pass
```

### TypeScript (Frontend)

- Follow Airbnb style guide
- Use TypeScript strict mode
- Maximum line length: 100 characters
- Use JSDoc for complex functions
- Format with Prettier

Example:
```typescript
/**
 * Upload patient images to the backend
 * 
 * @param patientId - Patient identifier
 * @param images - Array of image files
 * @returns Upload result with image set ID
 */
async function uploadImages(
  patientId: string,
  images: File[]
): Promise<UploadResult> {
  // Implementation
}
```

## Testing Guidelines

### Unit Tests

- Test individual functions/components
- Mock external dependencies
- Use descriptive test names
- Aim for >80% code coverage

Example:
```python
def test_pigmentation_severity_classification():
    """Test that pigmentation severity is correctly classified"""
    # Arrange
    area = create_test_pigmentation_area(intensity=45)
    
    # Act
    severity = classify_severity(area)
    
    # Assert
    assert severity == "Medium"
```

### Property-Based Tests

- Test universal properties
- Use Hypothesis (Python) or fast-check (TypeScript)
- Run minimum 100 iterations
- Tag with `@pytest.mark.property`

Example:
```python
@given(image_set=valid_image_set_strategy())
@settings(max_examples=100)
@pytest.mark.property
def test_complete_pigmentation_detection(image_set):
    """
    Property 1: Complete Pigmentation Detection
    For any valid image set, all visible pigmentation should be detected
    
    Validates: Requirements 1.1
    """
    # Test implementation
    pass
```

### Integration Tests

- Test component interactions
- Use real dependencies when possible
- Test complete workflows
- Tag with `@pytest.mark.integration`

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Include parameter types and return types
- Reference related requirements
- Provide usage examples for complex functions

### API Documentation

- Use OpenAPI/Swagger annotations
- Document all endpoints
- Include request/response examples
- Document error responses

### User Documentation

- Update README.md for user-facing changes
- Update SETUP.md for setup changes
- Add screenshots for UI changes
- Keep documentation in sync with code

## CI/CD Pipeline

### Automated Checks

On every push/PR:
1. Linting (flake8, ESLint)
2. Unit tests
3. Code coverage
4. Docker build
5. Security scanning

### Property-Based Tests

Run nightly or with `[run-pbt]` in commit message:
```bash
git commit -m "feat: add detection [run-pbt]"
```

### Deployment

- `develop` branch: Auto-deploy to staging
- `main` branch: Manual deploy to production
- Tagged releases: Create GitHub release

## Security Guidelines

### Sensitive Data

- Never commit secrets or credentials
- Use environment variables
- Encrypt patient data at rest
- Use HTTPS for all communications

### Code Security

- Validate all user inputs
- Sanitize data before database queries
- Use parameterized queries
- Follow OWASP guidelines

### Dependency Management

- Keep dependencies up to date
- Review security advisories
- Use Dependabot for automated updates
- Audit dependencies regularly

## Performance Guidelines

### Backend

- Use async/await for I/O operations
- Implement caching for expensive operations
- Optimize database queries
- Profile code for bottlenecks

### Frontend

- Lazy load components
- Optimize bundle size
- Use React.memo for expensive renders
- Implement virtual scrolling for large lists

### 3D Rendering

- Use level-of-detail (LOD)
- Implement frustum culling
- Optimize shader performance
- Target 30+ FPS

## Getting Help

- Check existing documentation
- Search closed issues/PRs
- Ask in team chat
- Create a discussion thread

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for your contributions!
