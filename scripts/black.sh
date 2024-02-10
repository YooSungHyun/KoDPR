#!/bin/bash

# 현재 디렉토리에서 시작하여 모든 하위 디렉토리를 순회하며 .py 파일 찾기
# 찾은 .py 파일마다 black을 실행하여 포맷팅
find . -type d -name .venv -prune -o -type f -name "*.py" -print | while read file; do
    black --line-length=119 "$file"
done

echo "All Python files have been formatted with black."