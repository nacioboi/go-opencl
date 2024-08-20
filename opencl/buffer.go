package opencl

// #include "opencl.h"
import "C"

type MemFlags uint64

const (
	MemReadWrite MemFlags = C.CL_MEM_READ_WRITE
	MemWriteOnly          = C.CL_MEM_WRITE_ONLY
	MemReadOnly           = C.CL_MEM_READ_ONLY
	// ...
)

type BufferType uint8

const (
	Float32 BufferType = iota
	Float64
	Int8
	Int16
	Int32
	Int64
	U_Int8
	U_Int16
	U_Int32
	U_Int64
	S_Int128
	S_Int256
	S_Int512
	S_U_Int128
	S_U_Int256
	S_U_Int512
)

type Buffer struct {
	buffer C.cl_mem
	_t     BufferType
}

func createBuffer(context Context, flags []MemFlags, size uint64, t BufferType) (Buffer, error) {
	// AND together all flags
	flagBitField := uint64(0)
	for _, flag := range flags {
		flagBitField &= uint64(flag)
	}

	var errInt clError
	buffer := C.clCreateBuffer(
		context.context,
		C.cl_mem_flags(flagBitField),
		C.size_t(size),
		nil,
		(*C.cl_int)(&errInt),
	)
	if errInt != clSuccess {
		return Buffer{}, clErrorToError(errInt)
	}

	return Buffer{buffer, t}, nil
}

func (b Buffer) Size() uint64 {
	return uint64(C.sizeof_cl_mem)
}

func (b Buffer) Release() {
	C.clReleaseMemObject(b.buffer)
}
