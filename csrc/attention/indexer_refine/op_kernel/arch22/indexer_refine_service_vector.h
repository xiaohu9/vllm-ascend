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
 * \file indexer_refine_service_vector.h
 * \brief
 */
#ifndef INDEXER_REFINE_SERVICE_VECTOR_H
#define INDEXER_REFINE_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../indexer_refine_common.h"
#include "indexer_refine_vector.h"

namespace LIKernel {
using namespace IndexerRefineCommon;
using namespace IndexerRefineServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

// 主模板：Q_T必选，W_T可选（默认void），无论W_T传什么，默认weightsType=Q_T
template<typename Q_T, typename W_T = void>
struct IndexerRefineTypeTraits {
    using weightsType = Q_T;   // 默认：weightsType绑定Q_T
};

// 偏特化1：固定第二个参数W_T=float，Q_T保留泛型
template<typename Q_T>
struct IndexerRefineTypeTraits<Q_T, float> {
    using weightsType = float;  // W_T=float时，强制weightsType为float
};

template <typename LIT>
class IndexerRefineServiceVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    using W_T = typename IndexerRefineTypeTraits<Q_T,
                                         typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline IndexerRefineServiceVector(){};
    __aicore__ inline void ProcessVec(const IndexerRefineCommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct IndexerRefineCommon::ConstInfo &constInfo,
                                      const IndexerRefineTilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<int32_t> candidatesGm,
                                                GlobalTensor<int32_t> paBlockTableGm, GlobalTensor<K_T> paKeyGm,
                                                GlobalTensor<K_T> gatheredKeyGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void GatherCandidates(uint32_t rStart, uint32_t rEnd);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    // refine 新增:stage-0 gather(mask/true_pos)用的 GM tensor
    GlobalTensor<int32_t> candidatesGm_;   // [R, coarseCount] 候选位置
    GlobalTensor<int32_t> paBlockTableGm_; // [R, maxBlockNumPerBatch] PA block table
    GlobalTensor<K_T> paKeyGm_;            // 原始 PA key cache(stage-0 gather 读)
    GlobalTensor<K_T> gatheredKeyGm_;      // workspace gather 区(AIC matmul 读)
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    // refine 新增:candidates 行全量 + blockTable 行 + key 中转(stage-0 gather / mask / true_pos 共用)
    TBuf<TPosition::VECCALC> candsFullBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t candsLoadedBIdx_ = -1; // refine:candsFullUb_ 已加载的 request 行号(惰性加载去重)

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct IndexerRefineCommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);                                            // 32KB  extract
    // 68KB 在搬运cube核计算得到的结果和weight时，分成两块34KB，用于db；在mrgsort时，用作临时UB
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));    // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
    // refine:reduceOutBuf_ 扩到 4×s2BaseSize_ — 段3 放 scattered mask(uint8),段4 放 NEG_INF 向量
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 4 * sizeof(float));                          // 8KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));
    // refine:candidates 行全量(int32,粗筛宽度) + blockTable 行 + 单候选 key 中转(256B)
    uint32_t candsFullSize = constInfo_.kSeqSize + constInfo_.maxBlockNumPerBatch + 64; // int32 元素
    pipe->InitBuffer(candsFullBuf_, candsFullSize * sizeof(int32_t));

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    globalTopkNum_ = 0;

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);

    // step3. 初始化vec1ParamGm，是否进行LD的标志位设为-1(needFd=-1)
    // vec1ResIn32Gm = [aic, 2, s1BaseSize_, 16] int32
    // ws清零 [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, ......]
    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    outQueue_.FreeTensor(tmpBuff);
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
    // refine:pipe->Reset() 已释放主循环 buffer,LD 阶段 true_pos 需重新申请 candidates 行缓冲
    uint32_t candsFullSize = constInfo_.kSeqSize + constInfo_.maxBlockNumPerBatch + 64;
    pipe->InitBuffer(candsFullBuf_, candsFullSize * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::InitParams(const struct IndexerRefineCommon::ConstInfo &constInfo,
                                                 const IndexerRefineTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;

    // group ub 切分因子当前按照UB空间强制为16
    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void
IndexerRefineServiceVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                    GlobalTensor<float> vec1ResGm,
                                    GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                    GlobalTensor<int32_t> indiceOutGm, GlobalTensor<int32_t> candidatesGm,
                                    GlobalTensor<int32_t> paBlockTableGm, GlobalTensor<K_T> paKeyGm,
                                    GlobalTensor<K_T> gatheredKeyGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->candidatesGm_ = candidatesGm;
    this->paBlockTableGm_ = paBlockTableGm;
    this->paKeyGm_ = paKeyGm;
    this->gatheredKeyGm_ = gatheredKeyGm;
}

// refine stage-0 gather:把候选 key 从 PA cache 散列搬入 workspace TND 布局。
// 分核与生产同步对齐:只聚本块 AIC 的 bN2 范围覆盖的请求整行 [rStart, rEnd](含),
// 两 AIV(blockId_%2)按请求奇偶各聚一半 → 本块 AIC matmul 消费的 workspace 全部由本块
// AIV 产出,块内 syncV1C1(预置 PIPE_MTE3)握手即可闭环,无跨块依赖、无需 SyncAll
// (生产 lightning_indexer 主流水线只有事件标志,无 SyncAll)。
template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::GatherCandidates(uint32_t rStart, uint32_t rEnd)
{
    uint32_t bs = constInfo_.kCacheBlockSize;
    uint32_t coarseCount = constInfo_.kSeqSize;
    LocalTensor<int32_t> candsUb = candsFullBuf_.Get<int32_t>();
    LocalTensor<int32_t> btRowUb = candsUb[coarseCount];
    LocalTensor<K_T> keyTmpUb = candsUb[coarseCount + constInfo_.maxBlockNumPerBatch].template ReinterpretCast<K_T>();
    uint32_t candsRowBlocks = coarseCount * sizeof(int32_t) / 32;
    uint32_t keyLenBlocks = constInfo_.headDim * sizeof(K_T) / 32;

    for (uint32_t rIdx = rStart + (blockId_ % 2); rIdx <= rEnd; rIdx += 2) {
        // 候选行整行入 UB;blockTable 行宽可 <8(int32 的 32B 对齐下限)且行 stride 非 32B,
        // DataCopy 无法整行搬(旧实现 btRowBlocks=0 → 未初始化 UB → 脏 slot → 非法 GM 读,即 507015 根因),
        // 改每 request 一次逐项 scalar 读入 UB,候选循环内改读 UB(同 AIC KeyNd2NzForPA 惯用法)
        AscendC::DataCopy(candsUb, candidatesGm_[rIdx * coarseCount], candsRowBlocks);
        for (uint32_t bIdx = 0; bIdx < constInfo_.maxBlockNumPerBatch; bIdx++) {
            btRowUb.SetValue(bIdx, paBlockTableGm_.GetValue(rIdx * constInfo_.maxBlockNumPerBatch + bIdx));
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        for (uint32_t cIdx = 0; cIdx < coarseCount; cIdx++) {
            int32_t cand = candsUb.GetValue(cIdx);
            if (cand < 0) {
                continue; // 无效候选:分数将置 -inf 沉底,无需 gather 其 key
            }
            uint32_t blockIdx = (uint32_t)cand / bs;
            uint32_t offsetInBlock = (uint32_t)cand % bs;
            int32_t base = btRowUb.GetValue(blockIdx);
            uint32_t slot = (base >= 0) ? (uint32_t)base * bs + offsetInBlock : 0;
            // 逐候选 256B:paKey[slot*Dh] → UB → gatheredKey[(r*coarse+c)*Dh]
            AscendC::DataCopy(keyTmpUb, paKeyGm_[(uint64_t)slot * constInfo_.headDim], keyLenBlocks);
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::DataCopy(gatheredKeyGm_[((uint64_t)rIdx * coarseCount + cIdx) * constInfo_.headDim], keyTmpUb,
                              keyLenBlocks);
        }
    }
    // 本核 MTE3 写 workspace 全部完成;交接由 kernel 预置 CrossCoreSetFlag(syncV1C1, PIPE_MTE3) 承担
    // (标志的 PIPE = 产生数据的 PIPE,同 lightning_indexer_quant 的 syncV1C1/syncV0C1 于 MTE3)
    AscendC::PipeBarrier<PIPE_MTE3>();
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    IndexerRefineServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::ProcessVec(const IndexerRefineCommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    // refine:candidates 行全量(scattered mask / true_pos 用),按 request 惰性加载
    LocalTensor<int32_t> candsFullUb = candsFullBuf_.Get<int32_t>();
    if (candsLoadedBIdx_ != static_cast<int32_t>(info.bIdx)) {
        AscendC::DataCopy(candsFullUb, candidatesGm_[info.bIdx * constInfo_.kSeqSize],
                          constInfo_.kSeqSize * sizeof(int32_t) / 32);
        AscendC::PipeBarrier<PIPE_MTE2>();
        candsLoadedBIdx_ = static_cast<int32_t>(info.bIdx);
    }
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                int64_t mmGmAllOffet = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                       outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t weightGmAllOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;

                IndexerRefineServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset,
                                     procGnum, info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                IndexerRefineServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use IndexerRefineServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            IndexerRefineServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), IndexerRefineServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            // refine scattered mask:cand==-1 的列分数置 -inf → topk 沉底 → 尾部输出自动 -1。
            // dav_c220 vsel 仅接受 __ubuf__ half*,float Select 同样无法编译 → 算术替代:
            //   mask=(cand!=-1):cand≥-1 ⇒ t=cand+1≥0,mask=(t>=1)?1:0 = Mins(t,t,1)(单条,免 bit-mask)
            //   位级 (score_bits-NEG_INF)*mask+NEG_INF:mask=1→原分、mask=0→NEG_INF(0xFF800000)
            LocalTensor<int32_t> candsSeg = candsFullUb[cuBaseS2Idx];
            LocalTensor<float> negInfUb = reduceOutBuff[3 * cuS2LenVecAlign];
            Duplicate(negInfUb.template ReinterpretCast<int32_t>(), IndexerRefineServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> maskI32 = sortIndiceUb.template ReinterpretCast<int32_t>(); // 复用 sortIndiceUb 段,L437 前会被重写
            Adds(maskI32, candsSeg, static_cast<int32_t>(1), cuS2Len);
            Mins(maskI32, maskI32, static_cast<int32_t>(1), cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> scoreI32 = sortScoreUb.template ReinterpretCast<int32_t>();
            Subs(scoreI32, scoreI32, IndexerRefineServiceVec::NEG_INF, cuS2Len);
            Mul(scoreI32, scoreI32, maskI32, cuS2Len);
            Adds(scoreI32, scoreI32, IndexerRefineServiceVec::NEG_INF, cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                IndexerRefineServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                      cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                PipeBarrier<PIPE_V>();
                LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                IndexerRefineServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                        cuS2LenVecAlign, UbTmpSort);
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();
                // 缓存4块512或者S2结束, 需要进行精排
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];
                    // 前4块直接精排覆盖到globalTopkUb_
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else { // 后面缓存在 SortedBasicBlock_, 先精排, 再merge到globalTopkUb_
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            if (needCopyOutGm) {
                int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen = (constInfo_.sparseCount <= SPARSE_COUNT_4K)
                                ? constInfo_.sparseCount
                                : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb,
                     globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset], (offset /32));

                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    // refine true_pos:topk 列号(全局 s2 位置) → candidates 值(原始 key 位置)
                    // Gather 的 srcOffset 语义 = 字节偏移,故先把列号 ×4;输出到 outValueUb[2*offset] 段
                    LocalTensor<int32_t> truePosUb = outValueUb[2 * offset].template ReinterpretCast<int32_t>();
                    // refine true_pos guard:globalTopk init 的 (-inf,-1) 槽被 topk 选中时列号=-1,
                    // ×4 → 0xFFFFFFFC → 4GB 字节偏移越界读。留 (列号>=0) 掩码 + clamp 列号到 0,
                    // Gather 后按掩码把无效槽写回 -1(有效候选→真实值, 无效槽→-1, 语义不变)。
                    // dav_c220 vsel 仅接受 __ubuf__ half*,int32 Select 无法编译 → 算术替代:
                    //   mask=(col>=0)?1:0 由 Mins(col,0)→Maxs(-1)→Adds(1) 生成(免 CompareScalar bit-mask),
                    //   dst=(src+1)*mask-1:mask=1→src、mask=0→-1。mask 暂存 value 段(offset>=copyLen)
                    LocalTensor<int32_t> maskI32 = outValueUb.template ReinterpretCast<int32_t>();
                    Mins(maskI32, idxULocal1, static_cast<int32_t>(0), copyLen);
                    Maxs(maskI32, maskI32, static_cast<int32_t>(-1), copyLen);
                    Adds(maskI32, maskI32, static_cast<int32_t>(1), copyLen);
                    PipeBarrier<PIPE_V>();
                    Maxs(idxULocal1, idxULocal1, static_cast<int32_t>(0), copyLen);
                    PipeBarrier<PIPE_V>();
                    Muls(idxULocal1, idxULocal1, 4, copyLen);
                    AscendC::Gather(truePosUb, candsFullUb, idxULocal1.template ReinterpretCast<uint32_t>(), 0,
                                    copyLen);
                    PipeBarrier<PIPE_MTE2>();
                    Adds(truePosUb, truePosUb, static_cast<int32_t>(1), copyLen);
                    Mul(truePosUb, truePosUb, maskI32, copyLen);
                    Adds(truePosUb, truePosUb, static_cast<int32_t>(-1), copyLen);
                    PipeBarrier<PIPE_V>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();

                    IndexerRefineServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx *
                                                         constInfo_.sparseCount + i * offset],
                                        truePosUb, copyLen);
                    outQueue_.FreeTensor(outValueUb);
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +       // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +       // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                // 写入头尾判断
                // [head, tail]
                // head: 与前面规约，与前后规约
                // tail: 与后面规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则 blockS2StartIdx_ != 0
                // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
                // 跟后面块做规约 写到1偏移  需要 + s1BaseSize_, BASE_TOPK*2
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                IndexerRefineServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                IndexerRefineServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void IndexerRefineServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        bn2Idx = vec1ParamGm.GetValue(wsInfoOffset + 5); // refine:true_pos 需按 request 定位 candidates 行
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
            bn2Idx = vec1ParamGm.GetValue(wsInfoOffset + 5);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // refine true_pos:LD 合并输出的列号 → candidates 值(原始 key 位置)
        LocalTensor<int32_t> candsFullUb = candsFullBuf_.Get<int32_t>();
        AscendC::DataCopy(candsFullUb, candidatesGm_[bn2Idx * constInfo_.kSeqSize],
                          constInfo_.kSeqSize * sizeof(int32_t) / 32);
        AscendC::PipeBarrier<PIPE_MTE2>();
        // refine true_pos guard:与 ProcessVec 同款 — (列号>=0) 掩码 + clamp, Gather 后无效槽写回 -1。
        // dav_c220 vsel 仅接受 __ubuf__ half*,int32 Select 无法编译 → 算术替代:
        //   mask=(col>=0)?1:0 由 Mins(col,0)→Maxs(-1)→Adds(1) 生成(免 CompareScalar bit-mask),
        //   dst=(src+1)*mask-1:mask=1→src、mask=0→-1。mask 暂存 tmpUb(原 validMask 段,此处空闲)
        LocalTensor<int32_t> maskI32 = tmpUb.template ReinterpretCast<int32_t>();
        Mins(maskI32, idxULocal1, static_cast<int32_t>(0), constInfo_.sparseCount);
        Maxs(maskI32, maskI32, static_cast<int32_t>(-1), constInfo_.sparseCount);
        Adds(maskI32, maskI32, static_cast<int32_t>(1), constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        Maxs(idxULocal1, idxULocal1, static_cast<int32_t>(0), constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        Muls(idxULocal1, idxULocal1, 4, constInfo_.sparseCount);
        LocalTensor<int32_t> truePosUb = outValueUb.template ReinterpretCast<int32_t>(); // value 段复用
        AscendC::Gather(truePosUb, candsFullUb, idxULocal1.template ReinterpretCast<uint32_t>(), 0,
                        constInfo_.sparseCount);
        AscendC::PipeBarrier<PIPE_MTE2>();
        Adds(truePosUb, truePosUb, static_cast<int32_t>(1), constInfo_.sparseCount);
        Mul(truePosUb, truePosUb, maskI32, constInfo_.sparseCount);
        Adds(truePosUb, truePosUb, static_cast<int32_t>(-1), constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(indiceOutGm[outOffset], truePosUb,
                    {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}
} // namespace LIKernel
#endif
