# Options for Sharing Your Code

Your repository is **6.9GB** - too large for standard Git. Here are better alternatives:

## Option 1: Git + Cloud Storage (Recommended)

**For code (small files):**
- Use Git/GitHub with the `.gitignore` I created
- Only track source code, configs, and documentation

**For large files (models, data, outputs):**
- Upload to Google Drive, Dropbox, or OneDrive
- Share download links in your README
- Or use GitHub Releases for files < 2GB

## Option 2: Create a Clean Archive

I've created a script to package only your code (excluding large files):

```bash
# This will create a clean zip with just code
./package_code.sh
```

Then share the zip file directly.

## Option 3: Use Git LFS (Git Large File Storage)

For files between 50MB-2GB:
```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
```

**Note:** Requires GitHub/GitLab account with LFS quota.

## Option 4: Direct File Sharing

1. **Create a code-only archive** (use the script below)
2. **Share via:**
   - Email (if < 25MB)
   - Google Drive / Dropbox link
   - WeTransfer (free, up to 2GB)
   - USB drive / external storage

## Quick Start: Package Your Code Now

Run this to create a clean code-only archive:

```bash
cd /Users/stanleyomondi/Desktop/Assign/proj1
tar -czf project1_code_only.tar.gz \
  --exclude='venv' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.mov' \
  --exclude='*.gif' \
  --exclude='pre-trained_models' \
  --exclude='**/experiment*/model.pt' \
  --exclude='**/experiment*/best_model.pt' \
  ml-equivariant-neural-rendering-main/ \
  *.py *.md *.txt *.json .gitignore 2>/dev/null
```

This creates `project1_code_only.tar.gz` with just your code.

