#!/usr/bin/env bash
# Fetch the Monaco road network into networks/. From the repo root:
#   source ~/venvs/spatialcov/bin/activate
#   bash get_networks.sh
# 1) Monaco: MoST scenario network (Codeca and Harri, 2017), GPL-3.0, pinned to a fixed commit.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)/networks"
mkdir -p "$DIR"
MOST_COMMIT=b29b2f65f1096a9c69a601ec62a724815cb4a43f
if [ ! -s "$DIR/most.net.xml" ]; then
  curl -fL -o "$DIR/most.net.xml" \
    "https://raw.githubusercontent.com/lcodeca/MoSTScenario/$MOST_COMMIT/scenario/in/most.net.xml"
fi
ls -la "$DIR"
