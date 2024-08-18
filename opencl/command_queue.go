package opencl

// #include "opencl.h"
import "C"
import (
	"errors"
	"fmt"
	"reflect"
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

	typeName := reflect.TypeOf(dataPtr).String()

	fmt.Printf("typeName: %s\n", typeName)

	var ptr unsafe.Pointer
	var dataLen uint64
	switch typeName {
	case "[]float32":
		p := dataPtr.([]float32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case "[]uint32":
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case "[][]uint32":
		p := dataPtr.([][]uint32)
		// Check if the length of 2nd dimension is 4.
		if len(p[0]) != 4 {
			return errors.New("Unexpected length of 2nd dimension. [][4]uint32 expected.")
		}
		// Flatten the 2D array to 1D.
		flat := make([]uint32, len(p)*4)
		for i := range p {
			copy(flat[i*4:], p[i][:])
		}
		dataLen = uint64(len(flat) * 4)
		ptr = unsafe.Pointer(&flat[0])
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

	typeName := reflect.TypeOf(dataPtr).String()

	var ptr unsafe.Pointer
	var dataLen uint64
	switch typeName {
	case "[]float32":
		p := dataPtr.([]float32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case "[]uint32":
		p := dataPtr.([]uint32)
		dataLen = uint64(len(p) * 4)
		ptr = unsafe.Pointer(&p[0])
	case "[][4]uint32":
		p := dataPtr.([][4]uint32)
		// Check if the length of 2nd dimension is 4.
		if len(p[0]) != 4 {
			return errors.New("Unexpected length of 2nd dimension. [][4]uint32 expected.")
		}
		// Flatten the 2D array to 1D.
		flat := make([]uint32, len(p)*4)
		for i := range p {
			copy(flat[i*4:], p[i][:])
		}
		dataLen = uint64(len(flat) * 4)
		ptr = unsafe.Pointer(&flat[0])
	default:
		return errors.New("Unexpected type for dataPtr")
	}

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
