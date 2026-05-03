#!/usr/bin/env bash
# Regenerate the gRPC Python bindings for the C2 service.
# Requires: pip install grpcio-tools
set -euo pipefail

cd "$(dirname "$0")"
python -m grpc_tools.protoc \
    -I=. \
    --python_out=. \
    --grpc_python_out=. \
    c2.proto

echo "regenerated: c2_pb2.py c2_pb2_grpc.py"
