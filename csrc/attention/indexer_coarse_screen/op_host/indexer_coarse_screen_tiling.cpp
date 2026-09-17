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
 * \file indexer_coarse_screen_tiling.cpp
 * \brief
 */

#include "indexer_coarse_screen_tiling.h"
#include "../op_kernel/indexer_coarse_screen_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {
namespace {
constexpr uint32_t Q_T_ELEM_SIZE = 2;      // bf16/f16
constexpr uint32_t I32_ELEM_SIZE = 4;      // int32
constexpr uint64_t GM_ALIGN_BYTES = 512;   // workspace 段对齐
} // namespace

// --------------------------IndexerCoarseScreenInfoParser类成员函数定义-------------------------------------
ge::graphStatus IndexerCoarseScreenInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor q_bar is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr, OP_LOGE(opName_, "Shape of tensor w_bar is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr, OP_LOGE(opName_, "Desc of tensor weights is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor key is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor key is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.candidatesOut.shape == nullptr,
               OP_LOGE(opName_, "Shape of tensor candidates is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.candidatesOut.desc == nullptr,
               OP_LOGE(opName_, "Desc of tensor candidates is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.aslkOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor aslk_out is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.aslkOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor aslk_out is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.coarseCount == nullptr, OP_LOGE(opName_, "attr coarse_count is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.hasWindow == nullptr, OP_LOGE(opName_, "attr has_window is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("IndexerCoarseScreen", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
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

void IndexerCoarseScreenInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void IndexerCoarseScreenInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QBAR_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QBAR_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WBAR_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WBAR_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    GetOptionalInputParaInfo();
}

void IndexerCoarseScreenInfoParser::GetOutputParaInfo()
{
    opParamInfo_.candidatesOut.desc = context_->GetOutputDesc(CANDIDATES_INDEX);
    opParamInfo_.candidatesOut.shape = context_->GetOutputShape(CANDIDATES_INDEX);
    opParamInfo_.aslkOut.desc = context_->GetOutputDesc(ASLK_OUT_INDEX);
    opParamInfo_.aslkOut.shape = context_->GetOutputShape(ASLK_OUT_INDEX);
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.coarseCount = attrs->GetAttrPointer<int32_t>(ATTR_COARSE_COUNT_INDEX);
    opParamInfo_.hasWindow = attrs->GetAttrPointer<int32_t>(ATTR_HAS_WINDOW_INDEX);
    opParamInfo_.groupSizeAttr = attrs->GetAttrPointer<int32_t>(ATTR_GROUP_SIZE_INDEX);
    if (opParamInfo_.coarseCount != nullptr) {
        OP_LOGI(context_->GetNodeName(), "coarse count is:%d", *opParamInfo_.coarseCount);
    }
    if (opParamInfo_.hasWindow != nullptr) {
        OP_LOGI(context_->GetNodeName(), "has_window is:%d", *opParamInfo_.hasWindow);
    }
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");
    // coarse_count:粗筛候选宽,仅支持 over-2K 骨架(2048, 8192] 且为 1024 整数倍。
    // 非 over-2K 的跨核 LD 归并已裁剪(生产恒 4096),<=2048 会丢拆核续核候选,故禁入。
    OP_CHECK_IF(((*opParamInfo_.coarseCount <= 2048) || (*opParamInfo_.coarseCount > static_cast<int32_t>(SPARSE_LIMIT))),
               OP_LOGE(opName_, "input attr coarse_count must be in (2048, 8192]."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((*opParamInfo_.coarseCount % 1024 != 0),
               OP_LOGE(opName_, "coarse_count must be an integer multiple of 1024."),
               return ge::GRAPH_FAILED);
    // 0=纯粗筛 1=窗口注入(生产) 2=dump 3=仅M1+dump(阶段二分) 4=M1+主pass+dump
    OP_CHECK_IF((*opParamInfo_.hasWindow < 0) || (*opParamInfo_.hasWindow > 4),
               OP_LOGE(opName_, "input attr has_window must be 0..4."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF((*opParamInfo_.groupSizeAttr <= 0) || (*opParamInfo_.groupSizeAttr > static_cast<int32_t>(GROUP_SIZE_LIMIT)),
               OP_LOGE(opName_, "input attr group_size must be in (0, 16]."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.candidatesOut.desc->GetDataType();

    OP_CHECK_IF((inputQType_ != inputKType_) || (inputQType_ != weightsType_),
            OP_LOGE(opName_, "The data types of the input query, key, and weights must be the same."),
            return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
               OP_LOGE(opName_, "The data types of the input query, key, and weights must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputType_ != ge::DT_INT32,
               OP_LOGE(opName_, "The data types of the output candidates must be int32."),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.aslkOut.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "The data types of the output aslk_out must be int32."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetAndCheckOptionalInput()
{
    // 固定组合:query=TND(key=PA_BSND),三个可选输入实际必填
    OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
               OP_LOGE(opName_, "input block_table must not be null for PA_BSND key"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
               OP_LOGE(opName_, "input actual_seq_lengths_key must not be null"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "input actual_seq_lengths_key data type only support int32"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.tensor == nullptr,
               OP_LOGE(opName_, "input actual_seq_lengths_query must not be null for TND query"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
               OP_LOGE(opName_, "input actual_seq_lengths_query data type only support int32"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::CheckShapeDim()
{
    // 固定组合: query=TND [N,H,D], key=PA_BSND [BlockNum,BlockSize,1,D], weights [N,H],
    //           row_weights [R,g], block_table [R,maxBlockNumPerBatch],
    //           aslq/aslk [R], out candidates [R,W'], aslk_out [R]
    OP_CHECK_IF((opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
               OP_LOGE(opName_, "the dim num of block_table's shape should be 2"), return ge::GRAPH_FAILED);

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t candidatesShapeDim = opParamInfo_.candidatesOut.shape->GetStorageShape().GetDimNum();
    uint32_t aslkOutShapeDim = opParamInfo_.aslkOut.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(kShapeDim != DIM_NUM_FOUR,
               OP_LOGE(opName_, "the dim num of key's shape should be %u, but now is %u", DIM_NUM_FOUR, kShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShapeDim != DIM_NUM_THREE,
               OP_LOGE(opName_, "the dim num of query's shape should be %u, but now is %u",
                DIM_NUM_THREE, qShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == DIM_NUM_TWO),
               OP_LOGE(opName_, "the dim num of weights's shape should be %u, but now is %u", DIM_NUM_TWO,
                weightsShapeDim),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(aslkOutShapeDim != DIM_NUM_TWO - 1,
               OP_LOGE(opName_, "the dim num of aslk_out's shape should be 1, but now is %u", aslkOutShapeDim),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetN1Size()
{
    // TND [N,H,D] → H 在 dim 1
    n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);

    OP_CHECK_IF(n1Size_ > QUERY_HEAD_NUM_LIMIT, OP_LOGE(opName_, "N1 is %u, but N1 must be no greater than %u.",
                n1Size_, QUERY_HEAD_NUM_LIMIT), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName) const
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OP_LOGE(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetAndCheckN2Size()
{
    // PA_BSND [BlockNum, BlockSize, N2, D] → N2 在 dim 2,恒 1
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1, OP_LOGE(opName_, "key shape[%u] is numhead, only support 1.", DIM_IDX_TWO),
    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE(opName_, "input query's head_num %u can not be a multiple of key's head_num %u.", n1Size_, n2Size_);
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetGroupSize()
{
    // M1 出核:g 由 attr 传入(编译期常量:decode=MTP g,prefill=PREFILL_GROUP)。
    // 窗口 = [aslk-(g-1), aslk+aslq差分),宽度 2g-1 必须用真实 g,不能用上限
    // (2026-09-17 实测:16 上限使窗口多扫 12 个池内位置,被 topk 排除的判"新"追加)。
    groupSize_ = static_cast<uint32_t>(*opParamInfo_.groupSizeAttr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetBatchSize()
{
    // TND:以 actual_seq_lengths_query 数组长度为 B 轴(R,请求数/组数)
    return GetActualSeqLenSize(bSize_, opParamInfo_.actualSeqLengthsQ.tensor, "input actual_seq_lengths_query");
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetHeadDim()
{
    // TND [N,H,D] → D 是第 2 维
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO);
    OP_CHECK_IF(headDim_ != HEAD_DIM_LIMIT, OP_LOGE(opName_, "input query's last dim head_dim only support 128."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_ONE));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
               OP_LOGE(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF((blockCount_ == 0),
                OP_LOGE(opName_, "input key's block_count cannot be 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::GetS2SizeForPageAttention()
{
    // 与生产 lightning_indexer 同源:s2Size = maxBlockNumPerBatch*blockSize(PA 扫描上限,
    // host 静态计算,捕获期常量);每请求真实域上界由 actual_seq_lengths_key 逐行掩码
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
              maxBlockNumPerBatch_, blockSize_, static_cast<int32_t>(s2Size_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexerCoarseScreenInfoParser::ValidateInputShapesMatch()
{
    /*
    固定 TND query + PA_BSND key:
    query [N,H,D], weights [N,H], row_weights [R,g],
    key [BlockNum,BlockSize,1,D], block_table [R, BatchMaxBlockNum],
    act_seq_k [R](粗筛域上界,绝对值), act_seq_q [R](累计),
    candidates [R,W'], aslk_out [R]
    */
    const int64_t reqNum = static_cast<int64_t>(bSize_);
    // -----------------------check BatchSize(R)-------------------
    OP_CHECK_IF((opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != reqNum) ||
                (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != reqNum) ||
                (opParamInfo_.candidatesOut.shape->GetStorageShape().GetDim(0) != reqNum) ||
                (opParamInfo_.aslkOut.shape->GetStorageShape().GetDim(0) != reqNum),
                OP_LOGE(opName_,
                    "actual_seq_lengths_key, block_table dim 0, row_weights dim 0, candidates dim 0 and aslk_out "
                    "dim 0 must all be same as batchSize %u.",
                    bSize_),
                return ge::GRAPH_FAILED);
    // -----------------------check N(总 query 行)-------------------
    int64_t qNsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qNsize),
                OP_LOGE(opName_, "TND case input query and weights dim 0 are %ld, %ld respectively, they must be same.",
                    qNsize, opParamInfo_.weights.shape->GetStorageShape().GetDim(0)),
                return ge::GRAPH_FAILED);
    // -----------------------check N1(H)-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != n1Size_),
               OP_LOGE(opName_, "input query, weight shape dim N1 must be same."), return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(DIM_IDX_THREE) != headDim_),
               OP_LOGE(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED);
    // -----------------------check 输出列宽 W'-------------------
    uint32_t outW = *opParamInfo_.coarseCount;
    if (*opParamInfo_.hasWindow != 0) { // 1 与 2 同宽(2 只写行首 7 词)
        outW = *opParamInfo_.coarseCount + 2 * groupSize_ - 1;
    }
    OP_CHECK_IF((opParamInfo_.candidatesOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE) != outW),
               OP_LOGE(opName_, "candidates shape last dim must be coarse_count(+windowG),"
                       "but now they are %u, %ld respectively.", outW,
                       opParamInfo_.candidatesOut.shape->GetStorageShape().GetDim(DIM_IDX_ONE)),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void IndexerCoarseScreenInfoParser::GenerateInfo(IndexerCoarseScreenTilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.platformInfo = platformInfo_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;

    liInfo.bSize = bSize_;
    liInfo.s2Size = s2Size_;
    liInfo.gSize = gSize_;         // H
    liInfo.groupSize = groupSize_; // g
    liInfo.windowG = 2 * groupSize_ - 1;
    liInfo.hasWindow = static_cast<uint32_t>(*opParamInfo_.hasWindow);
    liInfo.outW = *opParamInfo_.coarseCount + (liInfo.hasWindow != 0 ? liInfo.windowG : 0);

    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.weightsType = weightsType_;
    liInfo.outputType = outputType_;

    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    liInfo.pageAttentionFlag = true; // key 恒 PA_BSND
    liInfo.sparseCount = *opParamInfo_.coarseCount;

    liInfo.inputQLayout = DataLayout::TND;
    liInfo.inputKLayout = DataLayout::BnBsND;
}

ge::graphStatus IndexerCoarseScreenInfoParser::ParseAndCheck(IndexerCoarseScreenTilingInfo &liInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetGroupSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2SizeForPageAttention()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(liInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForIndexerCoarseScreen(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------IndexerCoarseScreenTiling类成员函数定义-----------------------
ge::graphStatus IndexerCoarseScreenTiling::DoTiling(IndexerCoarseScreenTilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    // 布局:|mm1ResGm(主 pass,双缓冲/核)|qBarGm(M1 输出)|wBarGm|proxyCumGm|candidatesWsGm(窗口模式中转)|
    // 仅 arch22(910b/910_93),无 DAV_3510 分支(def 未注册 950)
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<uint64_t>(M_BASE_SIZE) * S2_BASE_SIZE * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;

    const uint64_t reqNum = tilingInfo->bSize;
    const uint64_t headNum = tilingInfo->gSize;
    auto align512 = [](uint64_t bytes) -> uint64_t {
        return (bytes + GM_ALIGN_BYTES - 1) / GM_ALIGN_BYTES * GM_ALIGN_BYTES;
    };
    // M1 组均值代理输出(主 pass 的 query/weights 输入)
    workspaceSize += align512(reqNum * headNum * HEAD_DIM_LIMIT * Q_T_ELEM_SIZE);               // qBar [R,H,Dh]
    workspaceSize += align512(reqNum * headNum * Q_T_ELEM_SIZE);                                // wBar [R,H]
    // 主 pass TND s1 累计(每请求 1 行 proxy → [1..R],kernel 写入)
    workspaceSize += align512(reqNum * I32_ELEM_SIZE);                                          // proxyCum [R]
    // 窗口模式:主 pass 候选 [R,coarseCount] 先落 workspace,窗口阶段并集注入后写输出
    if (tilingInfo->hasWindow) {
        workspaceSize += align512(reqNum * tilingInfo->sparseCount * I32_ELEM_SIZE);            // candidatesWs
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = static_cast<size_t>(workspaceSize);

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_groupSize(tilingInfo->groupSize);
    tilingData_.set_windowG(tilingInfo->windowG);
    tilingData_.set_outW(tilingInfo->outW);
    tilingData_.set_hasWindow(tilingInfo->hasWindow);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // int DT_W_FLAG, DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, LAYOUT_T, KV_LAYOUT_T
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
ge::graphStatus TilingForIndexerCoarseScreen(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("IndexerCoarseScreen", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    IndexerCoarseScreenTilingInfo liInfo;
    IndexerCoarseScreenInfoParser coarseScreenInfoParser(context);
    if (coarseScreenInfoParser.ParseAndCheck(liInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    IndexerCoarseScreenTiling liTiling(context);
    return liTiling.DoTiling(&liInfo);
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(IndexerCoarseScreen)
    .Tiling(TilingForIndexerCoarseScreen)
    .TilingParse<IndexerCoarseScreenCompileInfo>(TilingPrepareForIndexerCoarseScreen);

} // namespace optiling
