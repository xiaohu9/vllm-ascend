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
 * \file indexer_coarse_screen_service_vector.h
 * \brief
 */
#ifndef INDEXER_COARSE_SCREEN_SERVICE_VECTOR_H
#define INDEXER_COARSE_SCREEN_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../indexer_coarse_screen_common.h"
#include "indexer_coarse_screen_vector.h"

namespace LIKernel {
using namespace IndexerCoarseScreenCommon;
using namespace IndexerCoarseScreenServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

// 主模板：Q_T必选，W_T可选（默认void），无论W_T传什么，默认weightsType=Q_T
template<typename Q_T, typename W_T = void>
struct IndexerCoarseScreenTypeTraits {
    using weightsType = Q_T;   // 默认：weightsType绑定Q_T
};

// 偏特化1：固定第二个参数W_T=float，Q_T保留泛型
template<typename Q_T>
struct IndexerCoarseScreenTypeTraits<Q_T, float> {
    using weightsType = float;  // W_T=float时，强制weightsType为float
};

template <typename LIT>
class IndexerCoarseScreenServiceVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    using W_T = typename IndexerCoarseScreenTypeTraits<Q_T,
                                         typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline IndexerCoarseScreenServiceVector(){};
    __aicore__ inline void ProcessVec(const IndexerCoarseScreenCommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct IndexerCoarseScreenCommon::ConstInfo &constInfo,
                                      const IndexerCoarseScreenTilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm);  // mm1 视图经 set 注入
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // ---- coarse_screen 专属:M1 组均值(AIV 全局预阶段)/ M2.5 窗口注入(AIV 全局后阶段) ----
    __aicore__ inline void InitCoarseGlobalTensor(GlobalTensor<uint32_t> callerSeqLenGmQ,
                                                  GlobalTensor<uint32_t> aslkGm,
                                                  GlobalTensor<int32_t> candidatesWsGm,
                                                  GlobalTensor<int32_t> candidatesOutGm,
                                                  GlobalTensor<int32_t> aslkOutGm,
                                                  GlobalTensor<int32_t> dbgMm1);
    __aicore__ inline void ProcessWindow(TPipe *pipe, uint32_t rBegin, uint32_t rEnd);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    // ---- coarse_screen 专属(caller 输入 + M1 输出 + 窗口阶段读写)----
    GlobalTensor<uint32_t> callerSeqLenGmQ_; // caller aslq [R] 累计(差分 own_tokens)
    GlobalTensor<uint32_t> aslkGm_;      // aslk [R] 粗筛域上界(绝对值)
    GlobalTensor<int32_t> dbgMm1Gm_;    // mm1Res core0 位视图(workspace 头,dump 用)
    GlobalTensor<int32_t> candidatesWsGm_; // 窗口模式主 pass 候选中转 [R,sparseCount]
    GlobalTensor<int32_t> candidatesOutGm_; // 输出 candidates [R,outW]
    GlobalTensor<int32_t> aslkOutGm_;    // 输出 aslk' [R]
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
    // 窗口阶段独立缓冲(pipe->Reset 后重新申请,先例 = InitLDBuffers)
    TBuf<TPosition::VECCALC> winCandBuf_;    // 候选行 int32 [sparseCount]
    TBuf<TPosition::VECCALC> winPosBuf_;     // 广播窗口位置 int32(判重)
    TBuf<TPosition::VECCALC> winMskBuf_;     // (cand-pos)^2 截断序列 int32(fold 归并)
    TBuf<TPosition::VECCALC> winOutBuf_;     // 输出行 int32 [outW]
    TBuf<TPosition::VECCALC> winAuxBuf_;     // [0,64) 窗口新增位置 + [64,...) aslk 行块


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

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct IndexerCoarseScreenCommon::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    // CopyOut 段需求(单 outValueUb,Extract 形态): 值[0,offset) + 索引[offset,2offset) = 2*offset floats。
    //   non-Over2K: offset=virTopK=2048, copyLen<=2048 → 4096;Over2K: offset=copyOff, copyNum=2 → 4096。
    //   reduceCacheBuf(groupInner_*s2BaseSize_+offsets) 更大,outQueue_ 按其取 max,富余充足。
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
    // 段1 [0,V) sort 分数(掩码后)、段2 [V,2V) sort 索引(cols);sort 硬件 merge-list 在
    // dst[dstSize+8] write-only,与生产 lightning_indexer 两段布局一致,5 段分配保留裕量。
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 5 * sizeof(float));                          // 10KB
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));

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
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitParams(const struct IndexerCoarseScreenCommon::ConstInfo &constInfo,
                                                 const IndexerCoarseScreenTilingData *__restrict tilingData)
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
IndexerCoarseScreenServiceVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    IndexerCoarseScreenServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessVec(const IndexerCoarseScreenCommon::RunInfo &info)
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
    // cuRealAcSeq: 当前基本块S1对应的AcSeq(粗筛域上界 upper[r],PA 布局恒绝对值)
    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
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

                IndexerCoarseScreenServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset,
                                     procGnum, info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                IndexerCoarseScreenServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use IndexerCoarseScreenServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            IndexerCoarseScreenServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            LocalTensor<int32_t> scoreI32 = sortScoreUb.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 生产原生掩码(与 lightning_indexer_service_vector.h:358-365 逐字节一致):
            // [0,V) 先全宽预填 -inf(部分块尾对齐),有效段用 Adds 覆写真实分数;
            // 索引段 [V,2V) 部分块尾补 -1,有效段 = 全局逻辑位置(0-based)。
            // 无效位置(score=-inf)在归并中沉底,输出尾部空槽 = InitSortOutBuf 预填的 -1。
            Duplicate(scoreI32, IndexerCoarseScreenServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            // 进 sort 前统一同步:reduceOutBuff 写(V/MTE)全部落定后才被排序读取。
            AscendC::PipeBarrier<PIPE_ALL>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            // over-2K(coarseCount=4096>2048)恒走 SortAll + 原生单次 MergeSort(virTopK):
            //   原生 lightning_indexer 生产同款(2026-09-16 起,替换 v11 双累积 —— 该形态
            //   从未 NPU 验证且多 chunk 实测 fault,详见分支内注释)。
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                if (constInfo_.isSparseCountOver2K) {
                    // 2026-09-16 对齐原生生产形态:单次 MergeSort(mrgDstNum=virTopK)。
                    //   v11 双累积(2×2048)源于当年"3-segment 腐蚀"假设 —— 已被证伪(实为
                    //   tie 误判,见 indexer-refine-tie-root-cause 记忆),该形态仅 CPU 仿真背书、
                    //   从未 NPU 验证(refine 的 17/17 全部 sparseCount=2048 非 over-2K);本算子
                    //   NPU 实测多 chunk(aslk>512)在 v11 路径 aicore MTE fault(A7/A8/A9 复现,
                    //   单 chunk 全过)。原生 lightning_indexer 的 over-2K 即单次归并(生产代码):
                    //   SortAll(chunk) + MergeSort(acc, virTopK, chunk, len, tmpUb_)。
                    SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
                    PipeBarrier<PIPE_V>();
                    LocalTensor<float> ubTmpSort = tmpUb_;
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK,
                                            reduceOutBuff, cuS2LenVecAlign, ubTmpSort);
                } else if (cuS2LenVecAlign == s2BaseSize_) {
                    IndexerCoarseScreenServiceVec::SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
                    PipeBarrier<PIPE_V>();
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                            cuS2LenVecAlign, tmpSortBuf);
                } else {
                    IndexerCoarseScreenServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                          cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                    PipeBarrier<PIPE_V>();
                    LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
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

            if (needCopyOutGm) {
                // 生产形态 CopyOut: Extract 分离 globalTopkUb_ 的 (value,index) 交错对,经 outQueue_
                //   EnQue/DeQue 缓冲后拷贝 idx 到输出。coarse_screen 无 value 输出,只拷贝索引。
                //   索引即 0-based 逻辑 key 位置(S2 扫描域 = 连续前缀),无候选列号间接层。
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
                    IndexerCoarseScreenServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx *
                                                                 constInfo_.sparseCount + i * offset],
                                        idxULocal1, copyLen);
                    outQueue_.FreeTensor(outValueUb);
                }
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

// --------------------------coarse_screen 专属:全局张量注册--------------------------
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitCoarseGlobalTensor(
    GlobalTensor<uint32_t> callerSeqLenGmQ, GlobalTensor<uint32_t> aslkGm,
    GlobalTensor<int32_t> candidatesWsGm, GlobalTensor<int32_t> candidatesOutGm,
    GlobalTensor<int32_t> aslkOutGm, GlobalTensor<int32_t> dbgMm1)
{
    callerSeqLenGmQ_ = callerSeqLenGmQ;
    aslkGm_ = aslkGm;
    candidatesWsGm_ = candidatesWsGm;
    candidatesOutGm_ = candidatesOutGm;
    aslkOutGm_ = aslkOutGm;
    dbgMm1Gm_ = dbgMm1;
}


// --------------------------M1 组均值代理(AIV 全局预阶段)--------------------------
// q_bar[r] = Σ_{i<own_r} rw[r,i]·query[cum_{r-1}+i] / Σ_{i<own_r} rw[r,i](fp32 累加,一次舍入 bf16)
// w_bar[r] 同式(caller weights);row_weights 全 1 时 = torch 均匀组均值 mean(1)。
// proxyCum[r] = r+1(主 pass TND s1 累计,每请求 1 行 proxy)。
// 收尾:PipeBarrier<MTE3>(本核 flush) + SyncAll(全 AIV 屏障)→ 所有 AIV 的 M1 写全局可见;
// ProcessMain 随后预置 syncV1C1×2,AIC 首个 matmul 的 CrossCoreWaitFlag 由此放行(§4.11 模式)。

// --------------------------M2.5 窗口注入(AIV 全局后阶段,hasWindow 门控)--------------------------
// 每请求一行:win = [aslk-(g-1), aslk+aslq差分) 与候选行求差集,新增位置按升序 append 到有效前缀后,
// 尾部 -1 补齐到 outW(与 torch _inject_local_window 逐位一致:有效数 = min(upper,c),topk 行内
// -1 恒在尾部);aslk'[r] = validC + nNew。hasWindow=0 时主 pass 已直写输出,仅算 aslk'=min(upper,c)。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessWindow(TPipe *pipe, uint32_t rBegin, uint32_t rEnd)
{
    // 行范围由 kernel 传入(= 同对 AIC 的 SplitCore 精确请求范围,偶 AIV 承担):
    // 行 r 的候选行由主 pass 的 AIV 2r CopyOut(MTE3)写出 —— 本分区让 AIV 2r 同时
    // 承担行 r 的窗口读(MTE2),同核 MTE3→MTE2 经 PipeBarrier 严格有序,消除跨对
    // GM 可见性时序(NPU 实测 3 行 S2 重复+S4 缺失双向判错 = 该读竞争的签名,
    // 与 qBar/proxyCum 竞态同族第三例)。
    // 主 pass 候选写(MTE3)全部完成后才可读回。
    // 2026-09-17 竞态终修:行 r 的候选行由本 AIV 主 pass CopyOut(MTE3)写出 —— 同核
    // 跨管道 MTE3→MTE2 必须用事件同步,PipeBarrier<PIPE_MTE3> 只排 MTE3 管道内部,
    // MTE2 读可越过未落地的 MTE3 写(NPU 实测 dedup 双向判错 +1dup/-1miss 即半写快照)。
    // M1 段同款先例:SetWaitFlag<MTE3_MTE2> 后 qBar 行为验证正确。
    PipeBarrier<PIPE_MTE3>();
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    SyncAll();
    // 独立窗口缓冲(pipe->Reset 释放主 pass 缓冲,先例 = InitLDBuffers)
    pipe->Reset();
    const uint32_t c = static_cast<uint32_t>(constInfo_.sparseCount);
    const uint32_t W = constInfo_.outW;
    const uint32_t cAlign = IndexerCoarseScreenCommon::Align<uint32_t>(c, 8);
    const uint32_t rowNum = static_cast<uint32_t>(constInfo_.batchSize);
    pipe->InitBuffer(winCandBuf_, c * sizeof(int32_t));
    pipe->InitBuffer(winPosBuf_, cAlign * sizeof(int32_t));
    pipe->InitBuffer(winMskBuf_, cAlign * sizeof(int32_t));
    pipe->InitBuffer(winOutBuf_, W * sizeof(int32_t));
    pipe->InitBuffer(winAuxBuf_, (64 + (rEnd - rBegin)) * sizeof(int32_t));

    LocalTensor<int32_t> candI32 = winCandBuf_.Get<int32_t>();
    LocalTensor<int32_t> posI32 = winPosBuf_.Get<int32_t>();
    LocalTensor<int32_t> mskI32 = winMskBuf_.Get<int32_t>();
    LocalTensor<int32_t> outI32 = winOutBuf_.Get<int32_t>();
    LocalTensor<int32_t> auxI32 = winAuxBuf_.Get<int32_t>();

    if (constInfo_.hasWindow == 2U) {
        // DEBUG dump(dedup 全链复刻判决):词16..22 = cand[up-3..up+3] 原位采样;
        // 词0..6 = 7 个窗口位置各自经【与真实 dedup 完全相同向量链】算出的 diffSum。
        // 数学期望:pos 在候选行 ⇒ diffSum = validC-1;不在 ⇒ = validC。
        LocalTensor<int32_t> dumpCand = winCandBuf_.Get<int32_t>();
        LocalTensor<int32_t> posI32 = winPosBuf_.Get<int32_t>();
        LocalTensor<int32_t> mskI32 = winMskBuf_.Get<int32_t>();
        for (uint32_t r = rBegin; r < rEnd; r++) {
            DataCopy(dumpCand, candidatesWsGm_[r * c], c);
            PipeBarrier<PIPE_MTE2>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            uint32_t up = aslkGm_.GetValue(r);
            uint32_t validC = IndexerCoarseScreenCommon::Min(up, c);
            uint32_t cumEnd = callerSeqLenGmQ_.GetValue(r);
            uint32_t cumBegin = (r == 0) ? 0U : callerSeqLenGmQ_.GetValue(r - 1);
            uint32_t own = cumEnd - cumBegin;
            int32_t winStart = static_cast<int32_t>(up) - static_cast<int32_t>(constInfo_.groupSize - 1);
            int32_t winEnd = static_cast<int32_t>(up) + static_cast<int32_t>(own);
            for (uint32_t j = 0; j < 7; j++) {
                outI32.SetValue(16 + j, dumpCand.GetValue(up - 3U + j));
            }
            // dedup 全链复刻(每位置独立,从头初始化,与真实 dedup 同一序列,
            // 含钳位先于平方的长 L 溢出修复 —— 复刻链必须与真实链逐算子同步)
            uint32_t nb = IndexerCoarseScreenCommon::CeilDiv(validC, 64U);
            uint32_t cols = (nb <= 16) ? 16 : (nb <= 32) ? 32 : 64;
            for (uint32_t j = 0; j < 7; j++) {
                int32_t pos = winStart + static_cast<int32_t>(j);
                if (pos < 0 || pos >= winEnd || validC == 0) {
                    outI32.SetValue(j, -12345); // 域外标记
                    continue;
                }
                Duplicate(mskI32, 0, 64 * cols);
                PipeBarrier<PIPE_V>();
                Duplicate(posI32, pos, validC);
                PipeBarrier<PIPE_V>();
                Sub(mskI32, dumpCand, posI32, validC);
                PipeBarrier<PIPE_V>();
                Mins(mskI32, mskI32, static_cast<int32_t>(1), validC);
                PipeBarrier<PIPE_V>();
                Maxs(mskI32, mskI32, static_cast<int32_t>(-1), validC);
                PipeBarrier<PIPE_V>();
                Mul(mskI32, mskI32, mskI32, validC);
                PipeBarrier<PIPE_V>();
                uint32_t now = 64;
                while (now > 1) {
                    now >>= 1;
                    Add(mskI32, mskI32, mskI32[now * cols], now * cols);
                    PipeBarrier<PIPE_V>();
                }
                uint32_t n2 = cols;
                while (n2 > 8) {
                    n2 >>= 1;
                    Add(mskI32, mskI32, mskI32[n2], n2);
                    PipeBarrier<PIPE_V>();
                }
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                int32_t diffSum = 0;
                for (int t = 0; t < 8; t++) {
                    diffSum += mskI32.GetValue(t);
                }
                outI32.SetValue(j, diffSum);
            }
            outI32.SetValue(15, static_cast<int32_t>(0xC0FFEE34));
            float v0 = dbgMm1Gm_.GetValue(0);
            outI32.SetValue(23, *reinterpret_cast<int32_t *>(&v0));
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(candidatesOutGm_[r * W], outI32, {1, static_cast<uint16_t>(28 * sizeof(int32_t)), 0, 0});
            auxI32.SetValue(0, static_cast<int32_t>(aslkGm_.GetValue(r)));
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(aslkOutGm_[r], auxI32, {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        }
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        return;
    }
    // 跨行 outI32 复用护栏(2026-09-20 隔离探针定界后终修):本行 DataCopyPad
    // (MTE3 读)与下一行 V 重填竞态 —— 融合式 SetWaitFlag 同 ID 连发实测无效
    // (L5: r0 抓拍到 r1 标量头刚写完的中间态),改分离式:SetFlag 紧跟每次
    // DataCopyPad 之后落 MTE3 标记,下一行首 WaitFlag 消费(pingpong 先例同源:
    // 事件一次一个,严格 set→wait→set 交替)。
    bool mte3MarkerPending = false;
    for (uint32_t r = rBegin; r < rEnd; r++) {
        if (mte3MarkerPending) {
            WaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            mte3MarkerPending = false;
        }
        uint32_t upper = aslkGm_.GetValue(r);
        uint32_t cumEnd = callerSeqLenGmQ_.GetValue(r);
        uint32_t cumBegin = (r == 0) ? 0U : callerSeqLenGmQ_.GetValue(r - 1);
        uint32_t own = cumEnd - cumBegin;
        uint32_t validC = IndexerCoarseScreenCommon::Min(upper, c);
        // 短序列直通(2026-09-20,与 ProcessMain 的行跳过配对):池域 [0,upper) 与
        // own [upper, upper+own) 恰好相邻,恒等行 ∪ 窗口 = 纯等差 [0, upper+own),
        // 去重/读回/组装全免。数值 = python 快路径的历史恒等输出(升序全前缀),
        // 与全量路径集合恒等(仅行序不同,N1 集合语义覆盖)。仅 hasWindow==1。
        if (constInfo_.hasWindow == 1U && upper <= c) {
            int32_t n = static_cast<int32_t>(upper) + static_cast<int32_t>(own);
            Duplicate(outI32, constInfo_.INVALID_IDX, W);
            PipeBarrier<PIPE_V>();
            // iota:标量头 32 + 向量倍增(偏移/长度 32 对齐)+ 标量尾
            uint32_t filled = 0;
            for (; filled < 32 && filled < static_cast<uint32_t>(n); filled++) {
                outI32.SetValue(filled, static_cast<int32_t>(filled));
            }
            // A3(910B) S 管道标量头 → V 管道倍增读:跨管道必须 S_V 事件
            // (PipeBarrier 只排本管道;同款教训见 V_S/MTE3_MTE2 先例)。
            if (filled > 0) {
                SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            }
            while (filled < static_cast<uint32_t>(n)) {
                uint32_t take = static_cast<uint32_t>(n) - filled;
                if (take > filled) take = filled;
                take &= ~31U;
                if (take == 0) break;
                Adds(outI32[filled], outI32[0], static_cast<int32_t>(filled), take);
                PipeBarrier<PIPE_V>();
                filled += take;
            }
            for (; filled < static_cast<uint32_t>(n); filled++) {
                outI32.SetValue(filled, static_cast<int32_t>(filled));
            }
            // A3: 标量尾(S)与向量段(V)都可能写了 outI32,MTE3 读前两者都要等
            // (正常路径同款成对先例:S_MTE3 + V_MTE3)。
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(candidatesOutGm_[r * W], outI32,
                        {1, static_cast<uint16_t>(W * sizeof(int32_t)), 0, 0});
            SetFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            mte3MarkerPending = true;
            auxI32.SetValue(64 + (r - rBegin), n);
            continue;
        }
        if (!constInfo_.hasWindow) {
            // 纯粗筛模式:主 pass 已直写输出(行距 = outW = coarseCount),仅算 aslk'
            auxI32.SetValue(64 + (r - rBegin), static_cast<int32_t>(validC));
            continue;
        }
        DataCopy(candI32, candidatesWsGm_[r * c], c);
        // 2026-09-17 竞态终修(A3 判决实证):dedup 首个 Sub(V) 可越过未完成的
        // MTE2 拷贝读 candI32 —— winStart 位置被判"缺席"追加重复(A3 的 61 即证)。
        // PipeBarrier<PIPE_MTE2> 只排 MTE2 内部;跨管道 MTE2→V 必须事件同步
        // (M1/复刻链同款先例:复刻链因先做 MTE2_S 等待而全对,真实链缺此事件)。
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 窗口收集:win = [winStart, winEnd) 内的有效位置,不在候选行的按升序 append。
        // validC > 0 时先去重:t = cand-pos 先钳位 [-1,1] 再平方(值域恰 {0,1})后求和,
        // 和 < validC ⇔ 存在精确相等(present)。钳位先于平方 ⇒ 对任意 int32 t 全域无溢出。
        // validC == 0(如新请求 prefill 组 0,upper=0):候选行全 -1,无去重可言,
        // 但窗口收集仍必须执行(torch 同款:该组 own tokens 全部进精筛域)。
        int32_t winStart = static_cast<int32_t>(upper) - static_cast<int32_t>(constInfo_.groupSize - 1);
        int32_t winEnd = static_cast<int32_t>(upper) + static_cast<int32_t>(own);
        uint32_t nNew = 0;
        for (uint32_t j = 0; j < constInfo_.windowG; j++) {
            int32_t pos = winStart + static_cast<int32_t>(j);
            if (pos < 0 || pos >= winEnd) {
                continue;
            }
            if (validC > 0) {
                // 全 int32 算术链(NPU 实测修复:float 链 Cast(int→float)+float
                // Mins 在 dav_c220 上 diffSum 恒 = validC,相等候选永不出 0 → 全误判;
                // refine scattered-mask 同源教训"dav_c220 float 链不可靠 → 算术替代")。
                // m_j = clamp(t_j,-1,1)^2:t==0⇔相等;t≠0 ⇒ 1。钳位必须先于平方 ——
                // |t| 上界是 seq_len(AIME 实测 65617),先平方在 |t|>46340 时 int32 溢出
                // 为负(Mins 钳不住)、|t|=65536 时 ≡0 假匹配,own tokens 全被误判
                // present 丢弃 → 长生成硬循环(2026-09-18 AIME 30 条中 2 条复现,
                // 恰为仅有的 L>46341 两条)。两级对齐归并布局 [64,cols],
                // cols∈{16,32,64},向量偏移 now*cols*4 ≥ 64B;二级折到 8 元素标量收尾。
                uint32_t nb = IndexerCoarseScreenCommon::CeilDiv(validC, 64U);
                uint32_t cols = (nb <= 16) ? 16 : (nb <= 32) ? 32 : 64;
                Duplicate(mskI32, 0, 64 * cols);
                PipeBarrier<PIPE_V>();
                Duplicate(posI32, pos, validC);
                PipeBarrier<PIPE_V>();
                Sub(mskI32, candI32, posI32, validC);
                PipeBarrier<PIPE_V>();
                Mins(mskI32, mskI32, static_cast<int32_t>(1), validC);
                PipeBarrier<PIPE_V>();
                Maxs(mskI32, mskI32, static_cast<int32_t>(-1), validC);
                PipeBarrier<PIPE_V>();
                Mul(mskI32, mskI32, mskI32, validC);
                PipeBarrier<PIPE_V>();
                // 一级:64 行按列折叠(in-place 重叠加法同 DoReduce 先例)
                uint32_t now = 64;
                while (now > 1) {
                    now >>= 1;
                    Add(mskI32, mskI32, mskI32[now * cols], now * cols);
                    PipeBarrier<PIPE_V>();
                }
                // 二级:cols → 8 元素(偏移 n*4 ≥ 32B)
                uint32_t n2 = cols;
                while (n2 > 8) {
                    n2 >>= 1;
                    Add(mskI32, mskI32, mskI32[n2], n2);
                    PipeBarrier<PIPE_V>();
                }
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                int32_t diffSum = 0;
                for (int t = 0; t < 8; t++) {
                    diffSum += mskI32.GetValue(t);
                }
                // diffSum = 不等于 pos 的候选个数;present ⇔ 存在相等 ⇔ diffSum < validC(纯整比)
                if (diffSum < static_cast<int32_t>(validC)) {
                    continue; // present:窗口位置已在候选行
                }
            }
            auxI32.SetValue(nNew, pos);
            nNew++;
        }

        // 组装输出行:[0,c)=候选行原序(含 -1 尾),[validC, validC+nNew)=窗口新增(升序),[c,W)=-1
        // 管线顺序(NPU 实测修复:UB→UB DataCopy 走 MTE2 管道,与 S 管道补丁、MTE3 落盘
        // 竞态 → 补丁被覆写/行内容混杂):V 预填 -1 → V_MTE2 → MTE2 直读 GM 候选行进
        // outI32 → MTE2_V → V_S → S 补丁 → S_MTE3 + V_MTE3 → MTE3 落盘,逐段显式配对。
        Duplicate(outI32, constInfo_.INVALID_IDX, W);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopy(outI32, candidatesWsGm_[r * c], c);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        if (nNew > 0) {
            for (uint32_t k = 0; k < nNew; k++) {
                outI32.SetValue(validC + k, auxI32.GetValue(k));
            }
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(candidatesOutGm_[r * W], outI32, {1, static_cast<uint16_t>(W * sizeof(int32_t)), 0, 0});
        SetFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        mte3MarkerPending = true;
        auxI32.SetValue(64 + (r - rBegin), static_cast<int32_t>(validC + nNew));
    }
    if (rEnd > rBegin) {
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(aslkOutGm_[rBegin], auxI32[64],
                    {1, static_cast<uint16_t>((rEnd - rBegin) * sizeof(int32_t)), 0, 0});
    }
}
} // namespace LIKernel
#endif
