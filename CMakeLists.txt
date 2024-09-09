file(GLOB_RECURSE SRCS *.cu *.h *.hpp *.cpp)

if(NOT BUILD_SHARED_LIBS)
  cuda_add_library(bmlib ${SRCS})
else()
  cuda_add_library(bmlib SHARED ${SRCS})
endif()
target_link_libraries(bmlib benchmark::benchmark pthread)
cuda_add_cublas_to_target(bmlib)
