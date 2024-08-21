package opencl

// #include "opencl.h"
import "C"
import (
	"errors"
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

	var p_size uint64
	var ptr unsafe.Pointer

	switch buffer.t {
	case Int8:
		p := dataPtr.([]int8)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int16:
		p := dataPtr.([]int16)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int32:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int64:
		p := dataPtr.([]int64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int8:
		p := dataPtr.([]uint8)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int16:
		p := dataPtr.([]uint16)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int32:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int64:
		p := dataPtr.([]uint64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])

	case Float32:
		p := dataPtr.([]float32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Float64:
		p := dataPtr.([]float64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])

	case V_Int64:
		p := dataPtr.([]int64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_Int128:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_Int256:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_Int512:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int64:
		p := dataPtr.([]uint64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int128:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int256:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int512:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])

	default:
		return errors.New("Unexpected type for dataPtr")
	}

	errInt := clError(C.clEnqueueReadBuffer(c.commandQueue,
		buffer.buffer,
		br,
		0,
		C.size_t(p_size*uint64(buffer.size_of_data_type)),
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

	var p_size uint64
	var ptr unsafe.Pointer

	switch buffer.t {
	case Int8:
		p := dataPtr.([]int8)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int16:
		p := dataPtr.([]int16)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int32:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Int64:
		p := dataPtr.([]int64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int8:
		p := dataPtr.([]uint8)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int16:
		p := dataPtr.([]uint16)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int32:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case U_Int64:
		p := dataPtr.([]uint64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])

	case Float32:
		p := dataPtr.([]float32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case Float64:
		p := dataPtr.([]float64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])

	case V_Int64:
		p := dataPtr.([]int64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_Int128:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_Int256:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_Int512:
		p := dataPtr.([]int32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int64:
		p := dataPtr.([]uint64)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int128:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int256:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])
	case V_U_Int512:
		p := dataPtr.([]uint32)
		p_size = uint64(len(p))
		ptr = unsafe.Pointer(&p[0])

	default:
		return errors.New("Unexpected type for dataPtr")
	}

	errInt := clError(C.clEnqueueWriteBuffer(c.commandQueue,
		buffer.buffer,
		br,
		0,
		C.size_t(p_size*uint64(buffer.size_of_data_type)),
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
