
# Version 2

## Features

- Vertex limit, [`maxClusterVertices`](include/nvcluster/nvcluster.h)
- Vertex underfill cost, [`costUnderfillVertices`](include/nvcluster/nvcluster.h)
- Implicit connection computation with [`itemVertices`](include/nvcluster/nvcluster.h)
- Shared library support in cmake, [`NVCLUSTER_BUILDER_SHARED`](CMakeLists.txt)
- Dynamic `parallelize` switch in [`nvcluster_ContextCreateInfo`](include/nvcluster/nvcluster.h)

## Code Quality

- Real C API, removing namespace, adding prefixes, symbol export
- Flattened API structs, avoiding pointer chains
- Removed macro based parallel for loops
- Internal use of std::span instead of raw pointers
- vec3f and AABB objects instead of inlined operations
- Fallback for missing libc++ parallel execution
