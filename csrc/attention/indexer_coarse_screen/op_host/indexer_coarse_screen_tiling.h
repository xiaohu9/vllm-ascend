/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_coarse_screen_tiling.h
 * \brief
 */

#ifndef INDEXER_COARSE_SCREEN_TILING_H_
#define INDEXER_COARSE_SCREEN_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"

namespace optiling {
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    BnBsND = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index(与 kernel 形参顺序一致: q_bar,w_bar,key,actual_seq_q,actual_seq_k,block_table)
constexpr uint32_t QBAR_INDEX = 0;
constexpr uint32_t WBAR_INDEX = 1;
constexpr uint32_t KEY_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
// Outputs Index
constexpr uint32_t CANDIDATES_INDEX = 0;
constexpr uint32_t ASLK_OUT_INDEX = 1;
// Attributes Index
constexpr uint32_t ATTR_COARSE_COUNT_INDEX = 0;
constexpr uint32_t ATTR_HAS_WINDOW_INDEX = 1;
// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 8192;          // coarseCount 上限(over-2K 骨架 s1BaseSize 公式的硬件边界)
constexpr uint32_t QUERY_HEAD_NUM_LIMIT = 64;
constexpr uint32_t GROUP_SIZE_LIMIT = 16;        // 组内 query 数上限(与 torch 侧 _MAX_GROUP 一致)
constexpr int32_t WINDOW_ON = 1;                 // has_window 开窗口注入

// -----------算子TilingData定义---------------
// 字段语义(与 kernel InitTilingData 一一对应):
//   bSize=R(batchSize,aslq 长度)  gSize=H(query head num,matmul g 轴)
//   s2Size=PA 扫描上限=maxBlockNumPerBatch*blockSize(host 静态,捕获期常量)
//   sparseCount=coarseCount(输出候选宽,4096>2048 → over-2K 骨架)
//   groupSize=g(row_weights.shape[1],组内 query 数;窗口 = [aslk-(g-1), aslk+aslq差分))
//   windowG=2g-1(窗口并集最大宽度)  outW=输出列宽(coarseCount + hasWindow*windowG)
//   hasWindow=1 窗口注入(生产)/0 纯粗筛(transition/debug,输出宽=coarseCount)
BEGIN_TILING_DATA_DEF(IndexerCoarseScreenTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, groupSize)
TILING_DATA_FIELD_DEF(uint32_t, windowG)
TILING_DATA_FIELD_DEF(uint32_t, outW)
TILING_DATA_FIELD_DEF(uint32_t, hasWindow)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IndexerCoarseScreen, IndexerCoarseScreenTilingData)

// -----------算子CompileInfo定义-------------------
struct IndexerCoarseScreenCompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct LiParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingRequiredParaInfo rowWeights = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingRequiredParaInfo candidatesOut = {nullptr, nullptr};
    TilingRequiredParaInfo aslkOut = {nullptr, nullptr};

    const int32_t *coarseCount = nullptr;
    const int32_t *hasWindow = nullptr;
};

// -----------算子Tiling入参信息类---------------
class IndexerCoarseScreenTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LiParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;    // R(aslq 长度)
    int64_t s2Size = 0;    // PA 扫描上限 = maxBlockNumPerBatch*blockSize
    uint32_t gSize = 0;    // H(query head num)
    uint32_t groupSize = 0; // g(row_weights.shape[1])
    // PageAttention
    bool pageAttentionFlag = true;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Others Flag
    uint32_t sparseCount = 0; // coarseCount(输出候选宽)
    uint32_t windowG = 0;     // 2g-1
    uint32_t outW = 0;        // 输出列宽
    uint32_t hasWindow = 1;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType weightsType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout(固定 TND query + PA_BSND key)
    DataLayout inputQLayout = DataLayout::TND;
    DataLayout inputKLayout = DataLayout::BnBsND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class IndexerCoarseScreenInfoParser {
public:
    explicit IndexerCoarseScreenInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~IndexerCoarseScreenInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName) const;
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetGroupSize();
    void GenerateInfo(IndexerCoarseScreenTilingInfo &liInfo);
    ge::graphStatus ParseAndCheck(IndexerCoarseScreenTilingInfo &liInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    LiParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t groupSize_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_INT32;
};

// ---------------算子Tiling类---------------
class IndexerCoarseScreenTiling {
public:
    explicit IndexerCoarseScreenTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoTiling(IndexerCoarseScreenTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    IndexerCoarseScreenTilingData tilingData_;
};

} // namespace optiling
#endif // INDEXER_COARSE_SCREEN_TILING_H_
