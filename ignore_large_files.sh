#!/bin/bash

find . -type f -name "*.txt" -size +50M > large_files.txt
cat large_files.txt >> .gitignore
git add .gitignore
echo "Large files to gitignore complete"
