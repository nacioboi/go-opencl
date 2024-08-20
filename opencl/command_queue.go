package opencl

// #include "opencl.h"
import "C"
import (
	"errors"
	"fmt"
	"unsafe"
)

type CommandQueue struct {
	commandQueue C.cl_command_queue
}

func createCommandQueue(context Context, device Device) (CommandQueue, error) {
	var errInt clError
	queue := C.clCreateCommandQueue(
		context.context,
		device.deviceID,
		0,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return CommandQueue{}, clErrorToError(errInt)
	}

	return CommandQueue{queue}, nil
}

func (c CommandQueue) EnqueueNDRangeKernel(kernel Kernel, workDim uint32, globalWorkSize []uint64) error {
	errInt := clError(C.clEnqueueNDRangeKernel(c.commandQueue,
		kernel.kernel,
		C.cl_uint(workDim),
		nil,
		(*C.size_t)(&globalWorkSize[0]),
		nil, 0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) EnqueueReadBuffer(buffer Buffer, blockingRead bool, dataPtr interface{}) error {
	var br C.cl_bool
	if blockingRead {
		br = C.CL_TRUE
	} else {
		br = C.CL_FALSE
	}

	var ptr unsafe.Pointer
	var dataLen uint64
	switch buffer._t {
	case Float32:
		p := dataPtr.([]float32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case Float64:
		p := dataPtr.([]float64)
		dataLen = uint64(len(p) * 8)
		ptr = unsafe.Pointer(&p[0])
	case Int8:
		p := dataPtr.([]int8)
		dataLen = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int16:
		p := dataPtr.([]int16)
		dataLen = uint64(len(p) * 2)
		ptr = unsafe.Pointer(&p[0])
	case Int32:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case Int64:
		p := dataPtr.([]int64)
		dataLen = uint64(len(p) * 8)
		ptr = unsafe.Pointer(&p[0])
	case U_Int8:
		p := dataPtr.([]uint8)
		dataLen = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int16:
		p := dataPtr.([]uint16)
		dataLen = uint64(len(p) * 2)
		ptr = unsafe.Pointer(&p[0])
	case U_Int32:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case U_Int64:
		p := dataPtr.([]uint64)
		dataLen = uint64(len(p) * 8)
		ptr = unsafe.Pointer(&p[0])
	case S_Int128:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * (4 * 4))
		ptr = unsafe.Pointer(&p[0])
	case S_Int256:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * (4 * 8))
		ptr = unsafe.Pointer(&p[0])
	case S_Int512:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * (4 * 16))
		ptr = unsafe.Pointer(&p[0])
	case S_U_Int128:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * (4 * 4))
		ptr = unsafe.Pointer(&p[0])
	case S_U_Int256:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * (4 * 8))
		ptr = unsafe.Pointer(&p[0])
	case S_U_Int512:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * (4 * 16))
		ptr = unsafe.Pointer(&p[0])
	default:
		return errors.New("Unexpected type for dataPtr")
	}

	errInt := clError(C.clEnqueueReadBuffer(c.commandQueue,
		buffer.buffer,
		br,
		0,
		C.size_t(dataLen),
		ptr,
		0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) EnqueueWriteBuffer(buffer Buffer, blockingRead bool, dataPtr interface{}) error {
	var br C.cl_bool
	if blockingRead {
		br = C.CL_TRUE
	} else {
		br = C.CL_FALSE
	}

	var ptr unsafe.Pointer
	var dataLen uint64
	switch buffer._t {
	case Float32:
		p := dataPtr.([]float32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case Float64:
		p := dataPtr.([]float64)
		dataLen = uint64(len(p) * 8)
		ptr = unsafe.Pointer(&p[0])
	case Int8:
		p := dataPtr.([]int8)
		dataLen = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int16:
		p := dataPtr.([]int16)
		dataLen = uint64(len(p) * 2)
		ptr = unsafe.Pointer(&p[0])
	case Int32:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case Int64:
		p := dataPtr.([]int64)
		dataLen = uint64(len(p) * 8)
		ptr = unsafe.Pointer(&p[0])
	case U_Int8:
		p := dataPtr.([]uint8)
		dataLen = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int16:
		p := dataPtr.([]uint16)
		dataLen = uint64(len(p) * 2)
		ptr = unsafe.Pointer(&p[0])
	case U_Int32:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case U_Int64:
		p := dataPtr.([]uint64)
		dataLen = uint64(len(p) * 8)
		ptr = unsafe.Pointer(&p[0])
	case S_Int128:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * 16)
		ptr = unsafe.Pointer(&p[0])
	case S_Int256:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * 32)
		ptr = unsafe.Pointer(&p[0])
	case S_Int512:
		p := dataPtr.([]int32)
		dataLen = uint64(len(p) * 64)
		ptr = unsafe.Pointer(&p[0])
	case S_U_Int128:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 16)
		ptr = unsafe.Pointer(&p[0])
	case S_U_Int256:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 32)
		ptr = unsafe.Pointer(&p[0])
	case S_U_Int512:
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 64)
		ptr = unsafe.Pointer(&p[0])
	default:
		return errors.New("Unexpected type for dataPtr")
	}

	fmt.Printf("dataLen: %d\n", dataLen)
	fmt.Printf("ptr: %v\n", ptr)

	errInt := clError(C.clEnqueueWriteBuffer(c.commandQueue,
		buffer.buffer,
		br,
		0,
		C.size_t(dataLen),
		ptr,
		0, nil, nil))
	return clErrorToError(errInt)
}

func (c CommandQueue) Release() {
	C.clReleaseCommandQueue(c.commandQueue)
}

func (c CommandQueue) Flush() {
	C.clFlush(c.commandQueue)
}

func (c CommandQueue) Finish() {
	C.clFinish(c.commandQueue)
}
