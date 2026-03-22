#!/bin/bash
# Build script for the MASDR-RAG paper
# Usage: ./build.sh
# Requirements: pdflatex, bibtex (standard LaTeX installation)

set -e

echo "=== Building MASDR-RAG Paper ==="

# Step 1: First LaTeX pass
echo "[1/4] First LaTeX pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || {
    echo "WARNING: First pass had errors (expected for references)"
}

# Step 2: BibTeX pass
echo "[2/4] BibTeX pass..."
bibtex main > /dev/null 2>&1 || {
    echo "WARNING: BibTeX had warnings (check references.bib)"
}

# Step 3: Second LaTeX pass (resolve references)
echo "[3/4] Second LaTeX pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

# Step 4: Third LaTeX pass (final)
echo "[4/4] Final LaTeX pass..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

# Check result
if [ -f main.pdf ]; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Output: main.pdf"
    echo "Pages: $(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo 'unknown')"
    echo ""
    echo "NOTE: Fields marked [FILL: ...] in blue need to be replaced"
    echo "      with your actual data before submission."
else
    echo ""
    echo "=== FAILED ==="
    echo "Check main.log for errors"
    exit 1
fi
