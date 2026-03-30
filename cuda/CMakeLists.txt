add_library(kernels STATIC kernels.cu)
set_target_properties(kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(stress_interface SHARED stress_interface.cu)
set_target_properties(stress_interface PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(stress_interface PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)
target_link_libraries(stress_interface PUBLIC kernels)
