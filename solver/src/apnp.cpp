#include "apnp.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(apnp, m) {
  m.doc() = "Python binding for anisotropicPnP Solver implementation.";
  
  m.def("APnPSolver", &APnPSolver);
  m.def("APnPSolver2", &APnPSolver2);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}