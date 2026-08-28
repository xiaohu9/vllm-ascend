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
 * \file indexer_refine_tiling.cpp
 * \brief
 */

#include "indexer_refine_tiling.h"
#include "../op_kernel/indexer_refine_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {
// --------------------------IndexerRefineInfoParser类成员函数定义-------------------------------------
ge::graphStatus IndexerRefineInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr, OP_LOGE(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr, OP_LOGE(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.candidates.shape == nullptr, OP_LOGE(opName_, "Shape of tensor candidates is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.candidates.desc == nullptr, OP_LOGE(opName_, "Desc of tensor candidates is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOut == nullptr, OP_LOGE(opName_, "attr layout_query is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.layOutKey == nullptr, OP_LOGE(opName_, "attr layout_key is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.sparseCount == nullptr, OP_LOGE(opName_, "attr sparse_count is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("IndexerRefine", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND950)) {
        OP_LOGE(opName_, "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
               OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void IndexerRefineInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void IndexerRefineInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    opParamInfo_.candidates.desc = context_->GetInputDesc(CANDIDATES_INDEX);
    opParamInfo_.candidates.shape = context_->GetInputShape(CANDIDATES_INDEX);
    GetOptionalInputParaInfo();
}

void IndexerRefineInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(INDEXER_REFINE);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(INDEXER_REFINE);
}

ge::graphStatus IndexerRefineInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    if (opParamInfo_.layOut != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_query is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.sparseCount != nullptr) {
        OP_LOGI(context_->GetNodeName(), "refine count is:%d", *opParamInfo_.sparseCount);
    }
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");
    // refine 固定 query=TND(连续总行), key=PA_BSND(从 PA cache stage-0 gather)
    OP_CHECK_IF((std::string(opParamInfo_.layOut) != "TND"),
               OP_LOGE(opName_, "input attr layout_query only supported TND for refine."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((std::string(opParamInfo_.layOutKey) != "PA_BSND"),
               OP_LOGE(opName_, "input attr layout_key only supported PA_BSND for refine."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((*opParamInfo_.sparseCount > 0) && (*opParamInfo_.sparseCount <= SPARSE_LIMIT)),
               OP_LOGE(opName_, "input attr sparse_count must > 0 and <= 8192."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(((*opParamInfo_.sparseCount > 2048) && (*opParamInfo_.sparseCount % 1024 != 0)),
               OP_LOGE(opName_, "when sparse_count > 2048, sparse_count must be an integer multiple of 1024."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_);
    OP_CHECK_IF(!inDTypeAllEqual,
            OP_LOGE(opName_, "The data types of the input query and key must be the same."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
               OP_LOGE(opName_, "The data types of the input query, key must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND950) {
        OP_CHECK_IF((inputQType_ != weightsType_),
                OP_LOGE(opName_, "The data types of the input query, key, and weights must be the same."),
                return ge::GRAPH_FAILED);
    } else {
        if (weightsType_ != ge::DT_FLOAT) {
            OP_CHECK_IF((inputQType_ != weightsType_),
                    OP_LOGE(opName_, "The data types of the input query, key, and weights must be the same."),
                    return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(outputType_ != ge::DT_INT32,
               OP_LOGE(opName_, "The data types of the output sparse_indices must be int32."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND},
        {"TND", DataLayout::TND},
        {"PA_BSND", DataLayout::BnBsND}
    };

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second;
    }

    std::string layoutKey(opParamInfo_.layOutKey);
    auto itKey = layoutMap.find(layoutKey);
    if (itKey != layoutMap.end()) {
        kLayout_ = itKey->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::BnBsND) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_key is PA_BSND, input block_table must not be null"),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            opParamInfo_.actualSeqLengths.tensor == nullptr,
            OP_LOGE(opName_, "when layout_key is PA_BSND, input actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                   OP_LOGE(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_key is TND, input actual_seq_lengths_key must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
               opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
                   OP_LOGE(opName_, "input actual_seq_lengths_key data type only support int32"),
                   return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
                   opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "input actual_seq_lengths_key data type only support int32"),
               return ge::GRAPH_FAILED);
    if (qLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
                   OP_LOGE(opName_, "when layout_query is TND, input actual_seq_lengths_query must not be null"),
                   return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor != nullptr &&
                   opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "input actual_seq_lengths_query data type only support int32"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(kLayout_ != DataLayout::BnBsND && opParamInfo_.blockTable.tensor != nullptr,
                   OP_LOGE(opName_, "when key layout is not PA_BSND, input block_table must be null"),
                   return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::CheckShapeDim()
{
    // 固定组合: query=TND [T,H,D], key=PA_BSND [BlockNum,BlockSize,1,D], weights [T,H],
    //           candidates [R,coarseCount], out [T,1,refineCount], block_table [R,maxBlockNumPerBatch]
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                   (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
               OP_LOGE(opName_, "the dim num of block_table's shape should be 2"), return ge::GRAPH_FAILED);

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t candidatesShapeDim = opParamInfo_.candidates.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(kShapeDim != DIM_NUM_FOUR,
               OP_LOGE(opName_, "the dim num of key's shape should be %u, but now is %u", DIM_NUM_FOUR, kShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShapeDim != DIM_NUM_THREE,
               OP_LOGE(opName_, "the dim num of query's shape should be %u, but now is %u",
                DIM_NUM_THREE, qShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != DIM_NUM_THREE,
               OP_LOGE(opName_, "the dim num of sparse_indices's shape should be %u, but now is %u",
                DIM_NUM_THREE, outShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == DIM_NUM_TWO),
               OP_LOGE(opName_, "the dim num of weights's shape should be %u, but now is %u", DIM_NUM_TWO,
                weightsShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(candidatesShapeDim != DIM_NUM_TWO,
               OP_LOGE(opName_, "the dim num of candidates's shape should be %u, but now is %u", DIM_NUM_TWO,
                candidatesShapeDim),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);

    OP_CHECK_IF(n1Size_ > QUERY_HEAD_NUM_LIMIT, OP_LOGE(opName_, "N1 is %u, but N1 must be no greater than %u.",
                n1Size_, QUERY_HEAD_NUM_LIMIT), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName) const
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OP_LOGE(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1, OP_LOGE(opName_, "key shape[%u] is numhead, only support 1.", n2Index),
    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if ((qLayout_ == DataLayout::TND)) {
        return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
    } else { // BSND
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus IndexerRefineInfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (qLayout_) {
        case DataLayout::TND:
            // TND格式: [Total, N, D] -> D是第2维(索引2)
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND格式: [Batch, SeqLen, N, D] -> D是第3维(索引3)
            dIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT, OP_LOGE(opName_, "input query's last dim head_dim only support 128."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
               OP_LOGE(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF((blockCount_ == 0),
                OP_LOGE(opName_, "input key's block_count cannot be 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetS2SizeForPageAttention()
{
    // stage-0 gather 需要 block 大小 + block table 宽(PA 布局信息)
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    // refine 的 s2Size = coarseCount(候选集宽度),非 PA 总 token 数
    s2Size_ = opParamInfo_.candidates.shape->GetStorageShape().GetDim(1);
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_(coarseCount) is %d",
              maxBlockNumPerBatch_, blockSize_, static_cast<int32_t>(s2Size_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerRefineInfoParser::GetS2Size()
{
    // refine 的 key 恒为 PA_BSND, s2Size = coarseCount(candidates 宽度, 见 GetS2SizeForPageAttention)
    if (kLayout_ == DataLayout::BnBsND) {
        return GetS2SizeForPageAttention();
    }
    OP_LOGE(opName_, "input attr layout_key only supported PA_BSND for refine.");
    return ge::GRAPH_FAILED;
}

ge::graphStatus IndexerRefineInfoParser::ValidateInputShapesMatch()
{
    /*
    refine 固定 TND query + PA_BSND key:
    query [T,H,D],
    key [BlockNum,BlockSize,1,D],
    weight [T,H],
    candidates [R,coarseCount],
    block_table [R, BatchMaxBlockNum],
    act_seq_k [R], act_seq_q [R],
    out [T,1,refineCount]
    */
    // -----------------------check BatchSize(R)-------------------
    OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_) ||
                (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                (opParamInfo_.candidates.shape->GetStorageShape().GetDim(0) != bSize_),
                OP_LOGE(opName_,
                    "TND case input actual_seq_lengths_key, block_table dim 0, candidates dim 0 are %ld, %ld, %ld respectively, they must be same as batchSize %u.",
                    opParamInfo_.actualSeqLengths.tensor->GetShapeSize(),
                    opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0),
                    opParamInfo_.candidates.shape->GetStorageShape().GetDim(0), bSize_),
                return ge::GRAPH_FAILED);
    // -----------------------check T-------------------
    uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                OP_LOGE(opName_, "TND case input query, weights and sparse_indices dim 0 are %u, %ld, %ld respectively, they must be same.",
                    qTsize, opParamInfo_.weights.shape->GetStorageShape().GetDim(0),
                    opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    // -----------------------check N1(H)-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != n1Size_),
               OP_LOGE(opName_, "input query, weight shape dim N1 must be same."), return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE) != headDim_),
               OP_LOGE(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    // -----------------------check N2(恒 1)-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != n2Size_),
               OP_LOGE(opName_, "input query and output sparse_indices shape n2 dim must be same,"
                       "but now they are %u, %ld respectively.",
                       n2Size_, opParamInfo_.attenOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE)),
               return ge::GRAPH_FAILED);
    // -----------------------check coarseCount(candidates 宽)-------------------
    OP_CHECK_IF((opParamInfo_.candidates.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != s2Size_),
               OP_LOGE(opName_, "candidates shape last dim must be same as coarseCount,"
                       "but now they are %u, %ld respectively.", static_cast<uint32_t>(s2Size_),
                       opParamInfo_.candidates.shape->GetStorageShape().GetDim(DIM_IDX_ONE)),
               return ge::GRAPH_FAILED);
    // -----------------------check refineCount(输出 topk 宽)-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO) != *opParamInfo_.sparseCount),
               OP_LOGE(opName_, "output sparse_indices shape last dim must be same as attr sparse_count,"
                       "but now they are %u, %ld respectively.", *opParamInfo_.sparseCount,
                       opParamInfo_.attenOut.shape->GetStorageShape().GetDim(DIM_IDX_TWO)),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void IndexerRefineInfoParser::GenerateInfo(IndexerRefineTilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.platformInfo = platformInfo_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;

    liInfo.bSize = bSize_;
    liInfo.s1Size = s1Size_;
    liInfo.s2Size = s2Size_; // coarseCount
    liInfo.gSize = gSize_;   // H

    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.weightsType = weightsType_;
    liInfo.outputType = outputType_;

    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    liInfo.pageAttentionFlag = layOutKeyStr == "PA_BSND" ? true : false;
    liInfo.sparseCount = *opParamInfo_.sparseCount; // refineCount

    liInfo.inputQLayout = qLayout_;
    liInfo.inputKLayout = kLayout_;
}

ge::graphStatus IndexerRefineInfoParser::ParseAndCheck(IndexerRefineTilingInfo &liInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(liInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForIndexerRefine(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------IndexerRefineTiling类成员函数定义-----------------------
ge::graphStatus IndexerRefineTiling::DoTiling(IndexerRefineTilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    constexpr uint32_t HEAD_DIM = 128;                // 与 kernel HEAD_DIM 一致
    constexpr uint32_t GM_ALIGN_BYTES = 512;          // 与 kernel GM_ALIGN_BYTES 一致
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // stage-0 gather 候选 key 区(refine 新增): 前置头区 [R*coarseCount*Dh],与 kernel Init 布局一致
    uint64_t gatheredKeySize = static_cast<uint64_t>(tilingInfo->bSize) * tilingInfo->s2Size * HEAD_DIM * sizeof(uint16_t);
    workspaceSize += (gatheredKeySize + GM_ALIGN_BYTES - 1) / GM_ALIGN_BYTES * GM_ALIGN_BYTES;
    // 主流程需Workspace大小
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t s1BaseSize = 4;
        constexpr uint32_t s2BaseSize = 128;
        workspaceSize +=
            s1BaseSize * ((tilingInfo->s2Size + s2BaseSize - 1) / s2BaseSize) * s2BaseSize * sizeof(uint16_t) * aicNum;
    } else {
        constexpr uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        // Decode流程(LD)需要Workspace大小
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = static_cast<size_t>(workspaceSize);

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size); // coarseCount
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount); // refineCount
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // int DT_W_FLAG, DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t weightsType = static_cast<uint32_t>(tilingInfo->weightsType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t weightTypeFlag = (weightsType == ge::DT_FLOAT) ? 1 : 0;
    uint64_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout, weightTypeFlag);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);     // 1: batchmode模式

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForIndexerRefine(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IndexerRefine", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    IndexerRefineTilingInfo liInfo;
    IndexerRefineInfoParser IndexerRefineInfoParser(context);
    if (IndexerRefineInfoParser.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    IndexerRefineTiling liTiling(context);
    return liTiling.DoTiling(&liInfo);
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(IndexerRefine)
    .Tiling(TilingForIndexerRefine)
    .TilingParse<IndexerRefineCompileInfo>(TilingPrepareForIndexerRefine);

} // namespace optiling
