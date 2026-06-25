# AlltoAllAttnUpdateAllGather op_host CMake
# vllm-ascend csrc 风格 (对齐邻居 dispatch_ffn_combine / matmul_allreduce_add_rmsnorm):
#   - op_api/ 嵌套于 op_host/op_api/ (旧结构, 与邻居一致)
#   - ACLNNTYPE aclnn_inner 单项: 外层 aclnn* wrapper 手写, 内层 aclnnInner* 走 codegen
#   - 不传 OP_API_INDEPENDENT / OP_API_DIR (走 obj_func.cmake 默认旧结构分支)

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        allto_all_attn_update_all_gather_def.cpp
    )
endif()

# AICore 编译选项 (per-SoC)
#   --cce-auto-sync=off        : 关闭 CCE 自动同步 (MC2 自管同步)
#   -DAICORE_EXCEPTION_RESTART : 仅 910_93 (A3), AI Core 硬件异常时自动重启
if ("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910_93")
    add_ops_compile_options(
        OP_NAME AlltoAllAttnUpdateAllGather
        OPTIONS
            --cce-auto-sync=off
            -DAICORE_EXCEPTION_RESTART
            -Wno-deprecated-declarations
            -Werror
    )
elseif ("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910b")
    add_ops_compile_options(
        OP_NAME AlltoAllAttnUpdateAllGather
        OPTIONS
            --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE allto_all_attn_update_all_gather ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
