package opencl

// #include "opencl.h"
import "C"
import "errors"

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
	V_Int64
	V_Int128
	V_Int256
	V_Int512
	V_U_Int64
	V_U_Int128
	V_U_Int256
	V_U_Int512
)

type Buffer struct {
	buffer            C.cl_mem
	t                 BufferType
	size_of_data_type uint8
	num_elements      uint32
}

func createBuffer(context Context, flags []MemFlags, t BufferType, num_elements uint32) (Buffer, error) {
	// OR together all flags
	flagBitField := uint64(0)
	for _, flag := range flags {
		flagBitField |= uint64(flag)
	}

	var data_len_multiplier uint8
	switch t {
	case Int8:
	case U_Int8:
		data_len_multiplier = 1
	case Int16:
	case U_Int16:
		data_len_multiplier = 2
	case Int32:
	case U_Int32:
		data_len_multiplier = 4
	case Int64:
	case U_Int64:
		data_len_multiplier = 8

	// All vectors are 32bits for each item in the vector.
	case V_Int64:
	case V_Int128:
	case V_Int256:
	case V_Int512:
	case V_U_Int64:
	case V_U_Int128:
	case V_U_Int256:
	case V_U_Int512:
		data_len_multiplier = 4

	case Float32:
		data_len_multiplier = 4
	case Float64:
		data_len_multiplier = 8

	default:
		return Buffer{}, errors.New("Unexpected type for t")
	}

	size := uint64(num_elements) * uint64(data_len_multiplier)

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

	return Buffer{buffer, t, data_len_multiplier, num_elements}, nil
}

func (b Buffer) SizeOfDataType() uint64 {
	return uint64(b.size_of_data_type)
}

func (b Buffer) Release() {
	C.clReleaseMemObject(b.buffer)
}

func (b Buffer) SizeAllocated() uint64 {
	return uint64(b.size_of_data_type) * uint64(b.num_elements)
}
