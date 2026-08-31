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
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<int32_t> candidatesGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidatesGm_;   // [R, coarseCount] 候选位置(mask/true_pos 读)
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
    // candidates 行全量(mask / true_pos 共用)
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
    int32_t candsLoadedBIdx_ = -1;  // refine:candsFullUb_ 已加载的 request 行号(惰性加载,双 key)
    int32_t candsLoadedS2Idx_ = -1; // refine:candsFullUb_ 已加载的 S2 chunk 基址(惰性加载,双 key)

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
    // CopyOut 段需求(单 outValueUb,Extract 形态): 值[0,offset) + 索引[offset,2offset) = 2*offset floats。
    //   non-Over2K: offset=virTopK=2048, copyLen<=2048 → 4096;Over2K: offset=copyOff, copyNum=2 → 4096。
    //   reduceCacheBuf(groupInner_*s2BaseSize_+offsets) 更大,outQueue_ 按其取 max,富余充足。
    // 2026-08-30 修复: 默认 8192 floats 仅覆盖 refine<=2048;refine=4096(over2k 用例) 旧 DIAG 直出
    //   truePos/tmpUb 越界 → 输出 NEG_INF 位模式垃圾。按 2*offset 扩容(Extract dst 需求 =
    //   dstValue[0,offset)+dstIndex[0,offset),前版 4*offset 是过度分配)。
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    if (constInfo_.isSparseCountOver2K) {
        int64_t copyOff = (constInfo_.sparseCount <= SPARSE_COUNT_4K)
                              ? constInfo_.sparseCount
                              : constInfo_.sparseCount / 2;
        outNeedBufSize = 2 * copyOff * sizeof(float);
    }
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.sparseCount : BASE_TOPK;

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);                                            // 32KB  extract
    // 68KB 在搬运cube核计算得到的结果和weight时，分成两块34KB，用于db；在mrgsort时，用作临时UB
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));    // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
    // refine:reduceOutBuf_ 扩到 5×s2BaseSize_(v6 2026-08-31) — 段1 [0,V) sort 分数(掩码后)、
    // 段2 [V,2V) sort 索引(cols,只写一次)、段3 [2V,3V)+段4 [3V,4V)+段5 [4V,5V) mask/score 链
    // 专用 scratch(mask / (score-NEG_INF)*mask 积 / score-NEG_INF 各占一段,全程非原地,规避
    // count 模式 dst==src 的 src[j+1] 位移坑)。v5(4 段)把 mask 写进 [V,2V) 再重写 cols + 链尾
    // 二次覆写 [0,V) → 多 chunk 下 sort 输入被双重写击穿(输出值槽=候选索引)。v6 终写只落
    // [0,V)+[V,2V),identity 下与生产(lightning_indexer,2 段)逐位一致。sort 硬件 merge-list 在
    // dst[dstSize+8](≈[2V,3V) 内),write-only,scratch 复用无害;2026-08-30 的"段4 是 merge-list
    // 区必须预写 NEG_INF"为误诊,已移除。
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 5 * sizeof(float));                          // 10KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));
    // candidates 行 chunk 级(int32):主循环只读当前 S2 chunk(≤512 int32=2KB)。v7(2026-08-31)
    //   UB 超限修复 — 前版整行(16KB@c=4096)使 v6 prod 主循环 UB 总需求 197.5KB 超 A3 910B 的
    //   192KB → VEC 读写越界(507015)。整行载入移入 InitLDBuffers(LD 阶段,pipe->Reset() 后独立,
    //   不受影响);ProcessVec 按 (request, chunk) 双 key 惰性载入,见 ProcessVec。
    pipe->InitBuffer(candsFullBuf_, s2BaseSize_ * sizeof(int32_t));

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
    uint32_t candsFullSize = IndexerRefineCommon::Align<uint32_t>(constInfo_.kSeqSize, 8);
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
                                    GlobalTensor<int32_t> indiceOutGm, GlobalTensor<int32_t> candidatesGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->candidatesGm_ = candidatesGm;
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
    // refine:candidates 行 chunk 级(scattered mask / true_pos 用),按 (request, S2-chunk) 双 key
    //   惰性加载。v7(2026-08-31) 与 InitBuffers 配套:主循环不持整行,candsFullUb 只装当前 chunk。
    LocalTensor<int32_t> candsFullUb = candsFullBuf_.Get<int32_t>();
    int32_t cuS2LenChunk = cuBaseS2Idx + s2BaseSize_ >= info.actS2Size ? info.actS2Size - cuBaseS2Idx
                                                                       : s2BaseSize_;
    if (candsLoadedBIdx_ != static_cast<int32_t>(info.bIdx) ||
        candsLoadedS2Idx_ != cuBaseS2Idx) {
        // Level2 DataCopy count 单位为元素;count 按 8(int32=32B)对齐(部分尾块,读入同行的
        //   padding 元素无副作用)。chunkAligned≤s2BaseSize_=512=buffer 容量。
        int32_t chunkAligned = static_cast<int32_t>(IndexerRefineCommon::Align<uint32_t>(cuS2LenChunk, 8));
        AscendC::DataCopy(candsFullUb,
                          candidatesGm_[info.bIdx * constInfo_.kSeqSize + cuBaseS2Idx], chunkAligned);
        AscendC::PipeBarrier<PIPE_MTE2>();
        candsLoadedBIdx_ = static_cast<int32_t>(info.bIdx);
        candsLoadedS2Idx_ = cuBaseS2Idx;
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
            LocalTensor<int32_t> scoreI32 = sortScoreUb.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // [0,V) 先全宽预填 NEG_INF(部分块尾对齐,与生产同),数据写只有一次(链尾 Adds)。
            Duplicate(scoreI32, IndexerRefineServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            // refine scattered mask:cand==-1 的列分数置 -inf → topk 沉底 → 尾部输出自动 -1。
            // dav_c220 vsel 仅接受 __ubuf__ half*,float Select 同样无法编译 → 算术替代:
            //   mask=(cand!=-1):cand≥-1 ⇒ t=cand+1≥0,mask=(t>=1)?1:0 = Mins(t,1)(免 bit-mask)
            //   位级 (score_bits-NEG_INF)*mask+NEG_INF:mask=1→原分、mask=0→NEG_INF(0xFF800000)
            // v6(2026-08-31) 根因修复: 前版(v5)把 mask 写进 sort 索引槽 [V,2V) 再重写 cols,
            //   且链尾对 [0,V) 二次覆写 → 多 chunk 下 sort 输入被双重写击穿(绕过实验坐实 mask
            //   链是唯一破坏源)。本版 [0,V) 数据只写一次(掩码后分数,Duplicate 仅预填尾对齐)、
            //   [V,2V) 只写一次(cols)、mask/(score-NEG_INF)*mask 积/score-NEG_INF 全部走
            //   [2V,5V) 专用 scratch 且每步非原地(规避 count 模式 dst==src 的 src[j+1] 位移坑)。
            //   identity(全有效 cand)下 [0,V)=原始分数位模式,[V,2V)=cols → 与生产逐位一致,
            //   sort 输入恢复生产形态。sort 硬件 merge-list 在 dst[dstSize+8](≈[2V,3V) 内),
            //   write-only,scratch 复用无害;2026-08-30 的"段4 merge-list 区必须预写 NEG_INF"
            //   为误诊,已移除。
            LocalTensor<int32_t> candsSeg = candsFullUb; // v7: chunk 级 buffer 基址,去 cuBaseS2Idx 偏移
            LocalTensor<int32_t> sMask = reduceOutBuff[2 * cuS2LenVecAlign].template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> sTmp = reduceOutBuff[3 * cuS2LenVecAlign].template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> sScr = reduceOutBuff[4 * cuS2LenVecAlign].template ReinterpretCast<int32_t>();
            Adds(sMask, candsSeg, static_cast<int32_t>(1), cuS2Len);                 // [2V] = cand+1
            PipeBarrier<PIPE_V>();
            Mins(sTmp, sMask, static_cast<int32_t>(1), cuS2Len);                     // [3V] = mask
            PipeBarrier<PIPE_V>();
            // dav_c220 缺 SubsImpl(接口声明在、实现缺失,CANN 9.1.0) → 补码等价: x - 0xFF800000 ≡ x + 0x00800000
            Adds(sScr, reduceOutInner.template ReinterpretCast<int32_t>(),
                 static_cast<int32_t>(-IndexerRefineServiceVec::NEG_INF), cuS2Len); // [4V] = score-NEG_INF
            PipeBarrier<PIPE_V>();
            Mul(sMask, sScr, sTmp, cuS2Len);                                         // [2V] = (score-NEG_INF)*mask
            PipeBarrier<PIPE_V>();
            Adds(scoreI32, sMask, IndexerRefineServiceVec::NEG_INF, cuS2Len);        // [0,V) 前 cuS2Len = 掩码后分数
            PipeBarrier<PIPE_V>();
            // [V,2V) sort 索引槽:只写一次(cols);部分块尾对齐补 -1。
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            // 进 sort 前统一同步:reduceOutBuff 写(V/MTE)全部落定后才被排序读取。
            AscendC::PipeBarrier<PIPE_ALL>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            // 2026-08-31 v5 修复(保留): Sort<float,true>+MrgBasicBlock 缓存路径(actS1Size<=4)在
            //   cuS2Len 全块(=512)时曾实证损坏 → 全块用例强制走 SortAll+MergeSort 生产路径
            //   (绕过实验证明该路径对 identity 全对);部分块(<=256,精确层用例 e0/c64/c256
            //   全过)保留缓存路径。v6 mask 链修复后此强制仍无害,保留待回归再评估。
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K || cuS2Len == s2BaseSize_) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                if (constInfo_.isSparseCountOver2K) {
                    // 2026-08-31 v11 根因修复: over2k 归并只用 2-list。旧路径
                    //   MergeSort(acc, mrgDstNum=virTopK=4096, chunk, ...) 的 mrgDstNum=4096>3072
                    //   进 MergeSort 3-segment 分支(4 路 MrgSort, elementLengths=[2048,1024,1024,chunk]),
                    //   该分支 ~50% 数据相关单相邻 swap —— over2k/prod_wide NPU 实证, 2-list 分支
                    //   全用例 100% 可靠。v10(拆 2×256)只改了 chunk 排序, 归并仍 3-segment → 无效,
                    //   且把 prod_wide 暴露成同源失败(其 v9 通过纯属种子运气, 位置随数据漂移)。
                    //   方案: 双累积 acc_U(排名1-2048)+acc_L(排名2049-4096)。每 chunk SortAll(512)
                    //   后两次 2-list 归并(mrgDstNum=virTopK/2=2048≤3072, 永不进 3-segment):
                    //     MergeSort(acc_U, 2048, chunk, len, tmpUb_): 被丢弃的 len 个最小对留在
                    //       tmpUb_[virTopK, virTopK+2*len)(MrgSort 全量输出, DataCopy 只拷回前 2048)
                    //     MergeSort(acc_L, 2048, tmpUb_[virTopK], len, tmpUb_[virTopK+2*len])
                    //   集合恒等: 各次丢弃尾之并 == 全部非 top-2048 元素 → acc_L==排名2049-4096;
                    //   输出 = acc_U+acc_L 拼接 == top-4096, 布局与旧单次 4096 归并逐位一致,
                    //   CopyOut(Extract 按 virTopK 宽读) 不变。CPU 逐位验证 3 形态全过:
                    //   plans/indexer_refine_v11_twopass_verify.py。virTopK/2 仍 >3072(sparseCount
                    //   >6144)需丢尾级联扩展, 当前未触达。
                    SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign); // 恢复整块 512 排序(probe prod 同款, 实证可靠)
                    PipeBarrier<PIPE_V>();
                    IndexerRefineServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK / 2,
                                            reduceOutBuff, cuS2LenVecAlign, tmpUb_);
                    IndexerRefineServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2 + virTopK], virTopK / 2,
                                            tmpUb_[virTopK], cuS2LenVecAlign, tmpUb_[virTopK + 2 * cuS2LenVecAlign]);
                } else if (cuS2LenVecAlign == s2BaseSize_) {
                    // 2026-08-31 v10 修复: 全块 512 排序拆 2×256。Sort32/MrgSort 硬件在 512 粒度
                    //   存在数据相关腐蚀(over2k 随机数据单相邻 swap 实证: 分数严格单调却被换序,
                    //   位置随数据漂移 102/252; Sort<float,true> 全块损坏同源, 见上 v5 注),
                    //   ≤256 实测可靠。此处: Sort<float,true> 各排 256 半, 经既有 MergeSort 分
                    //   两次并入累积(top-k 结合律 → 与整体 512 排序逐位一致), 永不构造 512 排序
                    //   列表、永不触发 4 路 128→512 归并层。
                    LocalTensor<uint32_t> idxU32 =
                        reduceOutBuff[s2BaseSize_].template ReinterpretCast<uint32_t>();
                    // 半A: 值 reduceOutBuff[0,256) + 索引 [512,768) → tmpSortBuf[0,512)
                    AscendC::Sort<float, true>(tmpSortBuf, reduceOutBuff, idxU32,
                                              tmpSortBuf[s2BaseSize_], cuS2LenVecAlign / 64);
                    AscendC::PipeBarrier<PIPE_V>();
                    // 半B: 值 reduceOutBuff[256,512) + 索引 [768,1024) → tmpSortBuf[512,1024)
                    AscendC::Sort<float, true>(tmpSortBuf[s2BaseSize_], reduceOutBuff[s2BaseSize_ / 2],
                                              idxU32[s2BaseSize_ / 2], tmpSortBuf[2 * s2BaseSize_],
                                              cuS2LenVecAlign / 64);
                    AscendC::PipeBarrier<PIPE_V>();
                    // 两半分别并入累积(半A 存于 tmpSortBuf, 故 tmpTensor 用 tmpUb_ 不可复用 tmpSortBuf)
                    IndexerRefineServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK,
                                            tmpSortBuf, cuS2LenVecAlign / 2, tmpUb_);
                    IndexerRefineServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK,
                                            tmpSortBuf[s2BaseSize_], cuS2LenVecAlign / 2, tmpUb_);
                } else {
                    IndexerRefineServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                          cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                    PipeBarrier<PIPE_V>();
                    LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                    IndexerRefineServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                            cuS2LenVecAlign, UbTmpSort);
                }
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
            // v8(2026-08-31) 根因修复: 原门控 isAllLoopEnd || isS2End 只在请求末 chunk 触发,
            //   而 S2 被 SplitCore 拆到多核时(probe prod 32块/24核、mid 8块/8核, 请求 S2 落多核),
            //   中间核的 accumulated globalTopkUb_ 从不写 ws → ProcessLD 链读到未初始化垃圾。
            //   正确门控 = 每核自身 s2 范围末 chunk(isLastS2InnerLoop, CalcS2LoopParams 的
            //   s2LoopEnd): 起始核(blockS2StartIdx_==0)→tail 槽「跟后面块做规约」,续核
            //   (blockS2StartIdx_!=0)→head 槽「跟前面块做规约」, 与下方 WS 偏移规则注释一致。
            //   单 chunk / 整请求单核路径 blockS2StartIdx_==0 && isS2End → needCopyOutGm 优先
            //   走 CopyOut, 不受影响(已验证 small/small_wide/over2k/prod_wide)。
            //   仿真 plans/indexer_refine_ws_emulation.py 复现: 现行门控 5 个多 chunk 用例全丢数据
            //   (与 NPU 0% 吻合), 修复后 9/9 全对。
            bool needCopyWsGm = info.isLastS2InnerLoop;

            if (needCopyOutGm) {
                // 生产形态 CopyOut(2026-08-31 v6 恢复): Extract 分离 globalTopkUb_ 的 (value,index)
                //   交错对,经 outQueue_ EnQue/DeQue 缓冲后拷贝 idx 到输出。取代 v3/v4 TEMP DIAG
                //   直出 raw 对(其 MTE3 直读 globalTopkUb_ 与下块 InitSortOutBuf 竞态,且污染输出
                //   格式)。refine 无 value 输出(returnValue=false),只拷贝索引。
                int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
                int64_t copyLen = (constInfo_.sparseCount <= SPARSE_COUNT_4K)
                                ? constInfo_.sparseCount
                                : constInfo_.sparseCount / 2;
                int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb,
                            globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset], offset / 32);
                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();
                    IndexerRefineServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx *
                                                                 constInfo_.sparseCount + i * offset],
                                        idxULocal1, copyLen);
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

        // 搬出(生产 lightning_indexer ProcessLD returnValue=false 同款):
        //   Extract 分离 (value,index) → 直拷 idx(列号)。无效候选已由评分链
        //   (service_cube.h:256 cand==-1 置 -inf)沉底, 尾部空槽 = InitSortOutBuf 的 -1。
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                    {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}
} // namespace LIKernel
#endif
