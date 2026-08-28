/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_refine_tiling.h
 * \brief
 */

#ifndef INDEXER_REFINE_TILING_H_
#define INDEXER_REFINE_TILING_H_

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
// Inputs Index(与 kernel 形参顺序一致: query,key,weights,candidates,actual_seq_q,actual_seq_k,block_table)
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t CANDIDATES_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
//Outputs Index
constexpr uint32_t INDEXER_REFINE = 0;
// Attributes Index
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
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
constexpr uint32_t SPARSE_LIMIT = 8192;          // refine 候选集宽度上限(coarseCount ≤ 8192)
constexpr uint32_t QUERY_HEAD_NUM_LIMIT = 64;
constexpr uint32_t COARSE_COUNT = 4096;          // 候选集宽度(coarse_screen 输出)
constexpr uint32_t REFINE_COUNT = 2048;          // 输出 topk 宽度(refine 输出)

// -----------算子TilingData定义---------------
// 字段语义(与 kernel InitTilingData 一一对应):
//   bSize=R(batchSize)  gSize=H(query head num)  s1Size=TND 总 query 行(非 BSND 场景不用)
//   s2Size=coarseCount(候选集宽度,candidates.shape[1])  sparseCount=refineCount(输出 topk 宽度)
//   blockSize=PA block 大小   maxBlockNumPerBatch=block_table 宽(stage-0 gather 用)
BEGIN_TILING_DATA_DEF(IndexerRefineTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IndexerRefine, IndexerRefineTilingData)

// -----------算子CompileInfo定义-------------------
struct IndexerRefineCompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct LiParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingRequiredParaInfo candidates = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingRequiredParaInfo attenOut = {nullptr, nullptr};

    const char *layOut = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *sparseCount = nullptr;
};

// -----------算子Tiling入参信息类---------------
class IndexerRefineTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LiParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0; // coarseCount(候选集宽度)
    uint32_t gSize = 0; // H(query head num)
    // PageAttention(stage-0 gather 用)
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Others Flag
    uint32_t sparseCount = 0; // refineCount(输出 topk 宽度)
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType weightsType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::BnBsND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class IndexerRefineInfoParser {
public:
    explicit IndexerRefineInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~IndexerRefineInfoParser() = default;

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
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    void GenerateInfo(IndexerRefineTilingInfo &liInfo);
    ge::graphStatus ParseAndCheck(IndexerRefineTilingInfo &liInfo);

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
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    // Layout
    DataLayout qLayout_ = DataLayout::BSND;
    DataLayout kLayout_ = DataLayout::BnBsND;
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
class IndexerRefineTiling {
public:
    explicit IndexerRefineTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoTiling(IndexerRefineTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    IndexerRefineTilingData tilingData_;
};

} // namespace optiling
#endif // INDEXER_REFINE_TILING_H_
