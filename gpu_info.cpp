#include "gpu_info.h"

int getL1CacheSize(int major, int minor) {
  if (major == 8) {
    // Ampere architecture (Compute Capability 8.x)
    return 128 * 1024;  // 128 KB per SM
  } else if (major == 7) {
    // Volta/Turing architecture (Compute Capability 7.x)
    return 64 * 1024;  // 64 KB per SM
  } else if (major == 6) {
    // Pascal architecture (Compute Capability 6.x)
    return 64 * 1024;  // 64 KB per SM
  } else if (major == 5) {
    // Maxwell architecture (Compute Capability 5.x)
    return 48 * 1024;  // 48 KB per SM
  } else if (major == 3) {
    // Kepler architecture (Compute Capability 3.x)
    return 16 * 1024;  // 16 KB or 48 KB per SM
  } else {
    // Unknown architecture or no L1 cache information available
    return 0;
  }
}

// 定义 Compute Capability 架构版本
enum class ComputeCapability {
  CC_3 = 3,
  CC_5 = 5,
  CC_6 = 6,
  CC_7 = 7,
  CC_8 = 8
};

// 定义一个结构体来存储 CUDA 核心和 Tensor 核心的数量
struct SM_Cores {
  int cudaCoresPerSM;
  int tensorCoresPerSM;
};

// 使用字典来存储不同架构的核心数量
std::map<ComputeCapability, SM_Cores> sm_cores_map = {
    {ComputeCapability::CC_3, {192, 0}},
    {ComputeCapability::CC_5, {128, 0}},
    {ComputeCapability::CC_6, {128, 0}},
    {ComputeCapability::CC_7, {64, 8}},
    {ComputeCapability::CC_8, {128, 4}}};

void GPUInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    ComputeCapability capability = static_cast<ComputeCapability>(prop.major);
    SM_Cores sm_cores = sm_cores_map[capability];

    int l1CacheSize = getL1CacheSize(prop.major, prop.minor);

    std::cout << "\n\n******************************************"
              << std::endl;                                     // 添加框框
    std::cout << "**\t " << prop.name << " \t**" << std::endl;  // 添加框框
    std::cout << "**\t  Memory: "
              << prop.totalGlobalMem / (1024 * 1024 * 1024.f) << "GB \t\t**"
              << std::endl;  // 添加框框
    std::cout << "**\t  Compute capability: " << prop.major << "." << prop.minor
              << " \t**" << std::endl;  // 添加框框
    std::cout << "**\t  L1 Cache Size: " << l1CacheSize / 1024
              << " KB(per SM)\t**" << std::endl;
    std::cout << "**\t  L2 Cache Size: " << prop.l2CacheSize / (1024 * 1024.f)
              << "MB \t\t**" << std::endl;  // 添加 L2 缓存大小
    std::cout << "**\t  Memory Bandwidth: "
              << (prop.memoryBusWidth / 8.0) * (prop.memoryClockRate * 1e-6) * 2
              << "GB/s **" << std::endl;
    std::cout << "**\t  CUDA Cores: "
              << prop.multiProcessorCount * sm_cores.cudaCoresPerSM << "\t\t**"
              << std::endl;
    std::cout << "**\t  Tensor Cores: "
              << prop.multiProcessorCount * sm_cores.tensorCoresPerSM
              << "\t\t**" << std::endl;  // 添加 Tensor 核心数量
    std::cout << "******************************************"
              << std::endl;  // 添加框框
    std::cout << std::endl;
  }
}

void copyRight() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-escape-sequence"

  // clang-format off

std::cout << "  /\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_" << std::endl;
std::cout << " |                     COPYRIGHT NOTICE                    | " << std::endl;
std::cout << " |      COPYRIGHT (C) ZHENNANC LTD. ALL RIGHTS RESERVED.   | " << std::endl;
std::cout << "  \\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/\\_/" << std::endl;
std::cout << "       |                                                           " << std::endl;
std::cout << "       |          _,-''-.               .-._                       " << std::endl;
std::cout << "       |     ,---':::::::\`            \\_::`.,...__               " << std::endl;  // 修正转义字符
std::cout << "      ~ \\   |::`.:::::::::::`.       ~    )::::::.--'              " << std::endl;  // 修正转义字符
std::cout << "         \\  |:_:::`.::::::::::`-.__~____,'::::(                    " << std::endl;  // 修正转义字符
std::cout << "  ~~~~    \\  \\```-:::`-.0:::::::::\\:::::::::~::\\       ~~~         " << std::endl;  // 修正转义字符
std::cout << "              )` `` `.::::::::::::|:~~:::::::::|      ~   ~~       " << std::endl;
std::cout << "  ~~        ,',' ` `` \\::::::::,-/:_:::::::~~:/                    " << std::endl;
std::cout << "          ,','/` ,' ` `\\::::::|,'   `::~~::::/  ~~        ~        " << std::endl;
std::cout << " ~       ( (  \\_ __,.-' \\:-:,-'.__.-':::::::'  ~    ~              " << std::endl;
std::cout << "     ~    \\`---''   __..--' `:::~::::::_:-'                        " << std::endl;
std::cout << "           `------''      ~~  \\::~~:::'                            " << std::endl;
std::cout << "        ~~   `--..__  ~   ~   |::_:-'                    ~~~       " << std::endl;
std::cout << "    ~ ~~     /:,'   `''---.,--':::\\          ~~       ~~           " << std::endl;
std::cout << "   ~         ``           (:::::::|  ~~~            ~~    ~        " << std::endl;
std::cout << " ~~      ~~             ~  \\:~~~:::             ~       ~~~        " << std::endl;
std::cout << "              ~     ~~~     \\:::~::          ~~~     ~             " << std::endl;
std::cout << "     ~~           ~~    ~~~  ::::::                     ~~         " << std::endl;
std::cout << "           ~~~                \\::::   ~~                           " << std::endl;
std::cout << "                        ~   ~~ `--'                                " << std::endl;

  // clang-format on

#pragma GCC diagnostic pop
}