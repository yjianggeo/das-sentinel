#!/bin/sh
set -eu

output_dir=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${1:?output directory required}")
sqlite_archive=sqlite-amalgamation-3530300.zip
sqlite_sha256=646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431
pysqlite_sha256=fbc69bfdc0cb43a5badd5403b126d5151371b5037e0397ba9802bb440c5b0021
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
cd "$build_dir"
curl --fail --location --output "$sqlite_archive" "https://sqlite.org/2026/$sqlite_archive"
echo "$sqlite_sha256  $sqlite_archive" | sha256sum --check --strict
python -m pip download --no-deps --no-binary=:all: pysqlite3==0.5.4
echo "$pysqlite_sha256  pysqlite3-0.5.4.tar.gz" | sha256sum --check --strict
unzip -q "$sqlite_archive"
tar -xzf pysqlite3-0.5.4.tar.gz
cp sqlite-amalgamation-3530300/sqlite3.c sqlite-amalgamation-3530300/sqlite3.h pysqlite3-0.5.4/
python -m pip install wheel
cd pysqlite3-0.5.4
python setup.py build_static bdist_wheel
mkdir -p "$output_dir"
cp dist/*.whl "$output_dir/"
