cmake_minimum_required(VERSION 3.10)
project(ive_demo)

set(CMAKE_CXX_STANDARD 11)

# 设置交叉编译
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER arm-himix200-linux-gcc)
set(CMAKE_CXX_COMPILER arm-himix200-linux-g++)

# 编译的时候根据实际mpp路径修改即可
set(MPP_DIR /home/marc/hi_sdk/Hi3516CV500_SDK_V2.0.2.0/smp/a7_linux/mpp/)
include_directories(${MPP_DIR}/include)
set(LIB_DIR ${MPP_DIR}/lib)

# 项目lib
set(PROJ_LIB_DIR ${CMAKE_SOURCE_DIR}/libs/)
include_directories(${PROJ_LIB_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -ldl")

# 链接库目录
link_directories(${PROJ_LIB_DIR} ${MPP_DIR}/lib)

add_executable(app src/main.cpp
                   src/common/sample_comm_vo.c
                   src/common/sample_comm_vi.c
                   src/common/sample_comm_isp.c
                   src/common/sample_comm_vpss.c
                   src/common/sample_comm_venc.c
                   src/common/sample_comm_sys.c)

target_link_libraries(app ive isp md hdmi mpi nnie securec sns_gc2053 tde VoiceEngine
                         upvqe pthread dl dnvqe _hiae _hiawb _hidrc _hidehaze _hildci _hicalcflicker
                         sns_imx415 sns_imx327 sns_imx335 sns_imx458 sns_imx307 sns_imx307_2l sns_imx327_2l sns_sc4210 sns_ov12870
                         sns_os05a sns_os08a10 sns_os04b10 sns_mn34220
                         opencv_world
                         )                   