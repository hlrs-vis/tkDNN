cmake_minimum_required(VERSION 3.15)
project(tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

option(ENABLE_OPENCV_CUDA_CONTRIB "Enable OpenCV CUDA Contrib" OFF )

find_package(CUDA 9.0 REQUIRED)
if (CUDA_FOUND)
    set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_script) # No suffix required
    execute_process(COMMAND "rm ${OUTPUTFILE}")
    set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/getCudaArch.cu)
    execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
    execute_process(COMMAND ${OUTPUTFILE}
            RESULT_VARIABLE CUDA_RETURN_CODE
            OUTPUT_VARIABLE ARCH)

    if(${CUDA_RETURN_CODE} EQUAL 0)
        set(CUDA_SUCCESS "TRUE")
    else()
        set(CUDA_SUCCESS "FALSE")
    endif()

    if (${CUDA_SUCCESS})
        message(STATUS "CUDA Architecture: ${ARCH}")
        message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
        message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
        message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")
        set(CUDA_NVCC_FLAGS "${ARCH}")
    else()
        message(WARNING ${ARCH})
    endif()
endif()

SET(CUDA_SEPARABLE_COMPILATION ON)

if(UNIX)
    if(CMAKE_BUILD_TYPE MATCHES Release)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-unused-variable -O3")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-unused-variable -g3")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32 -G -g)
    endif()
endif()

if(WIN32)
set(CMAKE_CXX_STANDARD 14)
    if(CMAKE_BUILD_TYPE MATCHES Release)
        set(CMAKE_CXX_FLAGS "/O2 /FS /EHsc /MD")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)
    endif()

    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_CXX_FLAGS "/Od /FS /EHsc /MDd")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32 -G -g)
    endif()
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)

# project specific flags
if(DEBUG)
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if(TKDNN_PATH)
    message("SET TKDNN_PATH:" ${TKDNN_PATH})
    add_definitions(-DTKDNN_PATH="${TKDNN_PATH}")
else()
    add_definitions(-DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(ENABLE_OPENCV_CUDA_CONTRIB ON)

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")


find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})


# compile
file(GLOB tkdnn_CUSRC "src/kernels/*.cu" "src/sorting.cu" "src/pluginsRT/*.cpp")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})
target_link_libraries(kernels ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES} ${CUDNN_LIBRARIES})



#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
message("Eigen DIR: " ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")
if(ENABLE_OPENCV_CUDA_CONTRIB)
    if (OpenCV_FOUND)
        find_package(OpenCV COMPONENTS cudawarping cudaarithm)
        if(OpenCV_cudawarping_FOUND AND OpenCV_cudaarithm_FOUND)
            add_compile_definitions(OPENCV_CUDACONTRIB)
            message("OpenCV Cuda Contrib modules found")
        else()
            message("OpenCV Cuda Contrib modules not found")
            set(ENABLE_OPENCV_CUDA_CONTRIB OFF)
        endif()
    endif()
endif()
# if(OpenCV_CUDA_VERSION)
#     add_compile_definitions(OPENCV_CUDACONTRIB)
# endif()

# gives problems in cross-compiling, probably malformed cmake config
find_package(yaml-cpp REQUIRED)

find_package(Boost REQUIRED)

#-------------------------------------------------------------------------------
# IDS Camera Library
#-------------------------------------------------------------------------------
set (CMAKE_SCRIPTS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/ids_opencv/_cmake_scripts" CACHE STRING "The path of the cmake scripts directory.")
include (${CMAKE_SCRIPTS_PATH}/cmake_detect_architecture.cmake)
detect_target_architecture (ARCH)
set (LIBRARY_NAME_VISION_API "ids_peak")
string (TOUPPER ${LIBRARY_NAME_VISION_API} LIBRARY_NAME_UPPER_VISION_API)
if (NOT TARGET ids_peak)
    file (TO_CMAKE_PATH "$ENV{IDS_PEAK_SDK_PATH}/api" ${LIBRARY_NAME_UPPER_VISION_API}_PACKAGE_DIR)

    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${${LIBRARY_NAME_UPPER_VISION_API}_PACKAGE_DIR}/cmake_finder")
    message (STATUS "[${PROJECT_NAME}] Will find IDS peak API library.. CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
    find_package (ids_peak REQUIRED)
endif ()

set (LIBRARY_NAME_LIBIMG "ids_peak_ipl")
string (TOUPPER ${LIBRARY_NAME_LIBIMG} LIBRARY_NAME_LIBIMG_UPPER)
if (NOT TARGET ids_peak_ipl)
    file (TO_CMAKE_PATH "$ENV{IDS_PEAK_SDK_PATH}/ipl" ${LIBRARY_NAME_LIBIMG_UPPER}_PACKAGE_DIR)

    set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${${LIBRARY_NAME_LIBIMG_UPPER}_PACKAGE_DIR}/cmake_finder")
    message (STATUS "[${PROJECT_NAME}] Will find IDS peak IPL library.. CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
    find_package (ids_peak_ipl REQUIRED)
endif ()

#if (EXISTS ${CMAKE_SOURCE_DIR}/lib/ids_opencv/CMakeLists.txt)
    add_subdirectory(lib/ids_opencv)
    set(with_ids true)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/ids_opencv)
    include_directories (ids_opencv_worker ${${LIBRARY_NAME_UPPER_VISION_API}_INCLUDE_DIR} ${${LIBRARY_NAME_LIBIMG_UPPER}_INCLUDE_DIR})
#endif()
#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS} ${CUDA_CUBLAS_LIBRARIES} ids_opencv_worker)

#static 
#add_library(tkDNN_static STATIC ${tkdnn_SRC})
#target_link_libraries(tkDNN_static ${tkdnn_LIBS})

# SMALL NETS
add_executable(test_simple tests/simple/test_simple.cpp)
target_link_libraries(test_simple tkDNN)

add_executable(test_mnist tests/mnist/test_mnist.cpp)
target_link_libraries(test_mnist tkDNN)

add_executable(test_mnistRT tests/mnist/test_mnistRT.cpp)
target_link_libraries(test_mnistRT tkDNN)

add_executable(test_imuodom tests/imuodom/imuodom.cpp)
target_link_libraries(test_imuodom tkDNN)

# DARKNET
file(GLOB darknet_SRC "tests/darknet/*.cpp")
foreach(test_SRC ${darknet_SRC})
    get_filename_component(test_NAME "${test_SRC}" NAME_WE)
    set(test_NAME test_${test_NAME})
    add_executable(${test_NAME} ${test_SRC})
    target_link_libraries(${test_NAME} tkDNN)
    install(TARGETS ${test_NAME} DESTINATION bin)
endforeach()

# MOBILENET
add_executable(test_mobilenetv2ssd tests/mobilenet/mobilenetv2ssd/mobilenetv2ssd.cpp)
target_link_libraries(test_mobilenetv2ssd tkDNN)

add_executable(test_bdd-mobilenetv2ssd tests/mobilenet/bdd-mobilenetv2ssd/bdd-mobilenetv2ssd.cpp)
target_link_libraries(test_bdd-mobilenetv2ssd tkDNN)

add_executable(test_mobilenetv2ssd512 tests/mobilenet/mobilenetv2ssd512/mobilenetv2ssd512.cpp)
target_link_libraries(test_mobilenetv2ssd512 tkDNN)

# BACKBONES
add_executable(test_resnet101 tests/backbones/resnet101/resnet101.cpp)
target_link_libraries(test_resnet101 tkDNN)

add_executable(test_dla34 tests/backbones/dla34/dla34.cpp)
target_link_libraries(test_dla34 tkDNN)

# CENTERNET
add_executable(test_resnet101_cnet tests/centernet/resnet101_cnet/resnet101_cnet.cpp)
target_link_libraries(test_resnet101_cnet tkDNN)

add_executable(test_dla34_cnet tests/centernet/dla34_cnet/dla34_cnet.cpp)
target_link_libraries(test_dla34_cnet tkDNN)

add_executable(test_dla34_cnet3d tests/centernet/dla34_cnet3d/dla34_cnet3d.cpp)
target_link_libraries(test_dla34_cnet3d tkDNN)

# CENTERTRACK

add_executable(test_dla34_ctrack tests/centertrack/dla34_ctrack/dla34_ctrack.cpp)
target_link_libraries(test_dla34_ctrack tkDNN)

# SHELFNET
add_executable(test_shelfnet tests/shelfnet/shelfnet.cpp)
target_link_libraries(test_shelfnet tkDNN)

add_executable(test_shelfnet_berkeley tests/shelfnet/shelfnet_berkeley.cpp)
target_link_libraries(test_shelfnet_berkeley tkDNN)

add_executable(test_shelfnet_mapillary tests/shelfnet/shelfnet_mapillary.cpp)
target_link_libraries(test_shelfnet_mapillary tkDNN)

# DEMOS
add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference tkDNN)

add_executable(map_demo demo/demo/map.cpp)
target_link_libraries(map_demo tkDNN)

add_executable(demo demo/demo/demo.cpp)
target_link_libraries(demo tkDNN)

if (with_ids)
add_executable(demoGeneral demo/demo/demoGeneral.cpp)
target_include_directories (demoGeneral
    PRIVATE ids_opencv_worker
    PRIVATE ${${LIBRARY_NAME_UPPER_VISION_API}_INCLUDE_DIR}
    PRIVATE ${${LIBRARY_NAME_LIBIMG_UPPER}_INCLUDE_DIR}
)
target_link_libraries(demoGeneral tkDNN ids_opencv_worker)
endif()

add_executable(demo3D demo/demo/demo3D.cpp)
target_link_libraries(demo3D tkDNN)

add_executable(demoTracker demo/demo/demoTracker.cpp)
target_link_libraries(demoTracker tkDNN)

add_executable(seg_demo demo/demo/seg_demo.cpp)
target_link_libraries(seg_demo tkDNN)

# KAFKA
find_package(CppKafka REQUIRED)

target_link_libraries(tkDNN CppKafka::cppkafka)

# TENSORFLOW
# set(TF_INCLUDE_DIR "/home/visadmin/.virtualenvs/opencv_cuda/lib/python3.8/site-packages/tensorflow/include")
# set(TF_LIB_DIR "/home/visadmin/.virtualenvs/opencv_cuda/lib/python3.8/site-packages/tensorflow")
# include_directories(${TF_INCLUDE_DIR})
# target_link_libraries(demoGeneral ${TF_LIB_DIR}/libtensorflow_framework.so)


# JSON
set(CMAKE_PREFIX_PATH "/usr/local/src/git/json/build")
find_package(nlohmann_json 3.2.0 REQUIRED HINTS "/usr/local/src/git/json/build")

target_link_libraries(tkDNN nlohmann_json::nlohmann_json)

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
#if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" 
#         CACHE PATH "default install path" FORCE)
#endif()
message("install dir:" ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY include/ DESTINATION include/)
# install(TARGETS tkDNN kernels DESTINATION lib)
install(TARGETS tkDNN  DESTINATION lib)
install(TARGETS test_simple test_mnist test_mnistRT test_rtinference demo map_demo demoGeneral DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" # source directory
        DESTINATION "share/tkDNN/cmake/" # target directory
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/" # source directory
        DESTINATION "share/tkDNN/tests" # target directory
)
