# Go OpenCL

>Forked from `github.com/MasterOfBinary/go-opencl` to add more creature comforts.

This is a very simple OpenCL wrapper for Go. To download, use `go get github.com/nacioboi/go-opencl`.

You'll need an OpenCL 2.0 library on all platforms except OS X. Download an SDK and copy its
`libOpenCL.a` file to `opencl/external/lib`. I recommend AMD APP SDK 3.0 or later, which can be downloaded
from [here](http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk).

To run it, make sure you have an SDK from Intel, NVIDIA, or AMD and a compatible
device. Then run with `go run main.go`:

