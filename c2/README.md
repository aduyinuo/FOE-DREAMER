# c2

Controller-side gRPC service used by the cyber-defense environment.

To regenerate the Python bindings from the protobuf schema:

```
pip install grpcio-tools
./generate_bindings.sh
```

This produces `c2_pb2.py` and `c2_pb2_grpc.py` in this directory; both
files are gitignored so they are regenerated per-checkout.
