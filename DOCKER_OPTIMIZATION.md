# Docker Image Size Optimization

## Changes Applied

### 1. **Optimized Multi-Stage Build**
- Changed builder WORKDIR from `/app` to `/install`
- Used `pip install --prefix=/install` to install only necessary packages
- Copy only `/install` to runtime (not entire `/usr/local`)
- Removed unnecessary pip metadata and cache

### 2. **Removed gosu Dependency**
- Eliminated `gosu` package (~1.5MB)
- Simplified entrypoint script
- Used native Docker `USER` directive instead

### 3. **Minimized Runtime Dependencies**
- Removed unused X11 libraries (`libsm6`, `libxrender1`, `libxext6`)
- Kept only essential libs for `opencv-python-headless`:
  - `libjpeg62-turbo`, `libpng16-16`, `libtiff6`, `libfreetype6`
  - `libgl1`, `libglib2.0-0`

### 4. **Layer Optimization**
- Merged RUN commands to reduce layers
- Cleaned up `__pycache__` and `.pyc` files in builder
- Used `--chown` in COPY to avoid extra chown layer
- Moved application code copy to last (better cache utilization)

### 5. **Enhanced .dockerignore**
- Added `Dockerfile*` and `docker-compose*.yml`
- Added all markdown files (`*.md`)
- Added documentation directories (`docs/`, `doc/`)
- Already excludes: `example/`, `outputs/`, `assets/`, `models/`, `test/`

### 6. **Python Bytecode Prevention**
- Set `PYTHONDONTWRITEBYTECODE=1` environment variable
- Clean up `.pyc`/`.pyo` files in both stages
- Remove `__pycache__` directories after copy

## Expected Size Reduction

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| gosu package | ~1.5MB | 0MB | ~1.5MB |
| Unused X11 libs | ~5MB | 0MB | ~5MB |
| pip metadata | ~10MB | ~2MB | ~8MB |
| Python bytecode | ~5MB | 0MB | ~5MB |
| **Total Estimated** | - | - | **~20-25MB** |

## Build Command

```bash
docker build -t panel-design:optimized .
```

## Verify Size

```bash
docker images panel-design:optimized
```

## Test Runtime

```bash
docker run --rm -p 8000:8000 panel-design:optimized
```

## Notes

- Image now runs as non-root user `appuser` (UID 1000)
- All runtime directories (`/app/outputs`, `/app/tmp`) are owned by `appuser`
- No privilege escalation needed at runtime
- Maintains same functionality with smaller footprint
