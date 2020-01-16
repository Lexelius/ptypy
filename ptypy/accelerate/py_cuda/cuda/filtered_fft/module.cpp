/*
<%
setup_pybind11(cfg)
cfg['sources'] = ['filtered_fft.cpp']
cfg['dependencies'] = ['errors.hpp', 'filtered_fft.hpp']
cfg['libraries'] = ['cufft_static', 'culibos', 'cudart_static']
cfg['parallel'] = True
%>
*/

/** This file contains the Python interface, exposed using PyBind11. */

#include <pybind11/pybind11.h>
#include "filtered_fft.hpp"


/** Wrapper class to expose to Python, taking size_t instead of all
 * the pointers, which should contain the addresses of the data.
 * For gpuarrays, the gpuarray.gpudata member can be cast to int
 * in Python and it will be the raw pointer address.
 * cudaStreams can be cast to int as well.
 * (this saves us a lot of type definition / conversion code with pybind11)
 *
 */
class FilteredFFTPython
{
public:
    FilteredFFTPython(int batches, bool symmetric, 
        std::size_t prefilt_ptr,
        std::size_t postfilt_ptr,
        std::size_t stream) 
    {
        fft_ = make_filtered(
            batches, 
            symmetric,
            reinterpret_cast<complex<float>*>(prefilt_ptr),
            reinterpret_cast<complex<float>*>(postfilt_ptr),
            reinterpret_cast<cudaStream_t>(stream)
        );
    }

    int getBatches() const { return fft_->getBatches(); }
    int getRows() const { return fft_->getRows(); }
    int getColumns() const { return fft_->getColumns(); }

    void fft(std::size_t in_ptr, std::size_t out_ptr)
    {
        fft_->fft(
            reinterpret_cast<complex<float>*>(in_ptr), 
            reinterpret_cast<complex<float>*>(out_ptr)
        );
    }

    void ifft(std::size_t in_ptr, std::size_t out_ptr)
    {
        fft_->ifft(
            reinterpret_cast<complex<float>*>(in_ptr), 
            reinterpret_cast<complex<float>*>(out_ptr)
        );
    }

    ~FilteredFFTPython() {
        delete fft_;
    }

private:
    FilteredFFT* fft_;
};


/////////////// Pybind11 Export Definition ///////////////

namespace py = pybind11;

#ifndef MODULE_NAME
#define MODULE_NAME filtered_fft 
#endif

PYBIND11_MODULE(MODULE_NAME, m) {
    m.doc() = "Filtered FFT for PtyPy";

    py::class_<FilteredFFTPython>(m, "FilteredFFT")
        .def(py::init<int, bool, std::size_t, std::size_t,std::size_t>(),
             py::arg("batches"), 
             py::arg("symmetricScaling"), 
             py::arg("prefilt"), 
             py::arg("postfilt"), 
             py::arg("stream")
        )
        .def("fft", &FilteredFFTPython::fft, 
             py::arg("input_ptr"), 
             py::arg("output_ptr")
        )
        .def("ifft", &FilteredFFTPython::ifft, 
             py::arg("input_ptr"), 
             py::arg("output_ptr")
        );
}

