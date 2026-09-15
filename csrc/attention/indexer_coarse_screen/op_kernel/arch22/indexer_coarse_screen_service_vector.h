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
                                                GlobalTensor<int32_t> indiceOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // ---- coarse_screen 专属:M1 组均值(AIV 全局预阶段)/ M2.5 窗口注入(AIV 全局后阶段) ----
    __aicore__ inline void InitCoarseGlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> callerWeightsGm,
                                                  GlobalTensor<Q_T> rowWeightsGm, GlobalTensor<uint32_t> callerSeqLenGmQ,
                                                  GlobalTensor<uint32_t> aslkGm, GlobalTensor<Q_T> qBarGm,
                                                  GlobalTensor<Q_T> wBarGm, GlobalTensor<int32_t> proxyCumGm,
                                                  GlobalTensor<int32_t> candidatesWsGm,
                                                  GlobalTensor<int32_t> candidatesOutGm, GlobalTensor<int32_t> aslkOutGm);
    __aicore__ inline void ProcessGroupMean();
    __aicore__ inline void ProcessWindow(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    // ---- coarse_screen 专属(caller 输入 + M1 输出 + 窗口阶段读写)----
    GlobalTensor<Q_T> queryGm_;          // caller query [N,H,Dh]
    GlobalTensor<Q_T> callerWeightsGm_;  // caller weights [N,H](原始 bf16/f16 存储视图,M1 读行用)
    GlobalTensor<Q_T> rowWeightsGm_;     // row_weights [R,g]
    GlobalTensor<uint32_t> callerSeqLenGmQ_; // caller aslq [R] 累计(差分 own_tokens)
    GlobalTensor<uint32_t> aslkGm_;      // aslk [R] 粗筛域上界(绝对值)
    GlobalTensor<Q_T> qBarGm_;           // q_bar [R,H,Dh](M1 输出)
    GlobalTensor<Q_T> wBarGm_;           // w_bar [R,H](M1 输出)
    GlobalTensor<int32_t> proxyCumGm_;  // [1..R](主 pass TND s1 累计)
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
    TBuf<TPosition::VECCALC> winCandF32Buf_; // 候选行 float [sparseCount](判重域)
    TBuf<TPosition::VECCALC> winPosF32Buf_;  // 广播窗口位置 float
    TBuf<TPosition::VECCALC> winMskF32Buf_;  // |cand-pos| 截断序列(fold 归并)
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
            // over-2K(coarseCount=4096>2048)恒走 SortAll + v11 双累积 2-list MergeSort:
            //   旧单次 4096 归并 mrgDstNum>3072 进 MergeSort 3-segment 分支(4 路 MrgSort),
            //   该分支 ~50% 数据相关单相邻 swap(over2k/prod_wide NPU 实证);2-list 分支
            //   全用例 100% 可靠(indexer_refine v11,2026-08-31 NPU 终验 17/17)。
            //   双累积: acc_U(排名1-2048)+acc_L(排名2049-4096),每 chunk SortAll(512) 后
            //   两次 mrgDstNum=2048 的归并,输出 = acc_U+acc_L 拼接 == top-4096。
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                if (constInfo_.isSparseCountOver2K) {
                    SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign); // 恢复整块 512 排序(probe prod 同款, 实证可靠)
                    PipeBarrier<PIPE_V>();
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK / 2,
                                            reduceOutBuff, cuS2LenVecAlign, tmpUb_);
                    // 2026-09-01 aarch64 原生工具链(严格模式)拒收 LocalTensor::operator[] 临时量作
                    //   非 const 左值引用形参(mrgSrc/tmpTensor): 先提命名变量。
                    LocalTensor<float> ubTail = tmpUb_[virTopK];
                    LocalTensor<float> ubScratch = tmpUb_[virTopK + 2 * cuS2LenVecAlign];
                    IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2 + virTopK], virTopK / 2,
                                            ubTail, cuS2LenVecAlign, ubScratch);
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
    GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> callerWeightsGm, GlobalTensor<Q_T> rowWeightsGm,
    GlobalTensor<uint32_t> callerSeqLenGmQ, GlobalTensor<uint32_t> aslkGm, GlobalTensor<Q_T> qBarGm,
    GlobalTensor<Q_T> wBarGm, GlobalTensor<int32_t> proxyCumGm, GlobalTensor<int32_t> candidatesWsGm,
    GlobalTensor<int32_t> candidatesOutGm, GlobalTensor<int32_t> aslkOutGm)
{
    queryGm_ = queryGm;
    callerWeightsGm_ = callerWeightsGm;
    rowWeightsGm_ = rowWeightsGm;
    callerSeqLenGmQ_ = callerSeqLenGmQ;
    aslkGm_ = aslkGm;
    qBarGm_ = qBarGm;
    wBarGm_ = wBarGm;
    proxyCumGm_ = proxyCumGm;
    candidatesWsGm_ = candidatesWsGm;
    candidatesOutGm_ = candidatesOutGm;
    aslkOutGm_ = aslkOutGm;
}

// --------------------------M1 组均值代理(AIV 全局预阶段)--------------------------
// q_bar[r] = Σ_{i<own_r} rw[r,i]·query[cum_{r-1}+i] / Σ_{i<own_r} rw[r,i](fp32 累加,一次舍入 bf16)
// w_bar[r] 同式(caller weights);row_weights 全 1 时 = torch 均匀组均值 mean(1)。
// proxyCum[r] = r+1(主 pass TND s1 累计,每请求 1 行 proxy)。
// 收尾:PipeBarrier<MTE3>(本核 flush) + SyncAll(全 AIV 屏障)→ 所有 AIV 的 M1 写全局可见;
// ProcessMain 随后预置 syncV1C1×2,AIC 首个 matmul 的 CrossCoreWaitFlag 由此放行(§4.11 模式)。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessGroupMean()
{
    const uint32_t qRowSize = static_cast<uint32_t>(constInfo_.headDim * constInfo_.gSize); // H*Dh
    const uint32_t hSize = static_cast<uint32_t>(constInfo_.gSize);
    const uint32_t rowNum = static_cast<uint32_t>(constInfo_.batchSize);
    const uint32_t aivNum = GetBlockNum() * 2;
    const uint32_t rowsPerAiv = IndexerCoarseScreenCommon::CeilDiv(rowNum, aivNum);
    const uint32_t rBegin = static_cast<uint32_t>(GetBlockIdx()) * rowsPerAiv;
    const uint32_t rEnd = IndexerCoarseScreenCommon::Min(rBegin + rowsPerAiv, rowNum);

    if (rBegin < rEnd) {
        // proxyCum:主 pass TND s1 累计 = [1..R];tmpUb 复用段,写完等 MTE3 读完成再让位 MTE2
        LocalTensor<int32_t> cumRow = tmpUb_.template ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(cumRow, static_cast<int32_t>(rBegin) + 1, 1, rEnd - rBegin);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(proxyCumGm_[rBegin], cumRow,
                    {1, static_cast<uint16_t>((rEnd - rBegin) * sizeof(int32_t)), 0, 0});
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);

        // UB 复用布局(H=64 时 68KB 内): rowQ bf16 [0,16KB) | rowF32 f32 [16KB,48KB)
        //   | rwF32 [48KB,+256B) | rwRow bf16 [+256B,+288B)
        LocalTensor<Q_T> rowQ = tmpUb_.template ReinterpretCast<Q_T>();
        LocalTensor<float> rowF32 = tmpUb_[qRowSize];
        LocalTensor<float> rwF32 = tmpUb_[2 * qRowSize];
        LocalTensor<Q_T> rwRow = tmpUb_.template ReinterpretCast<Q_T>()[4 * qRowSize + 2 * hSize];
        LocalTensor<float> acc = outQueue_.AllocTensor<float>();
        LocalTensor<float> accW = brcBuf_.Get<float>();
        for (uint32_t r = rBegin; r < rEnd; r++) {
            // row_weights 行(g bf16 不足 32B,DataCopyPad 装载)→ fp32,后续标量读
            DataCopyPad(rwRow, rowWeightsGm_[r * hSize],
                        {1, static_cast<uint16_t>(hSize * sizeof(Q_T)), 0, 0}, {false, 0, 0, 0});
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            Cast(rwF32, rwRow, RoundMode::CAST_NONE, hSize);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

            uint32_t cumEnd = callerSeqLenGmQ_.GetValue(r);
            uint32_t cumBegin = (r == 0) ? 0U : callerSeqLenGmQ_.GetValue(r - 1);
            uint32_t own = cumEnd - cumBegin;
            float sumRw = 0.0f;
            Duplicate(acc, 0.0f, qRowSize);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < own; i++) {
                float rw = rwF32.GetValue(i);
                sumRw += rw;
                DataCopy(rowQ, queryGm_[(cumBegin + i) * qRowSize], qRowSize);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(rowF32, rowQ, RoundMode::CAST_NONE, qRowSize);
                PipeBarrier<PIPE_V>();
                Muls(rowF32, rowF32, rw, qRowSize);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, rowF32, qRowSize);
                PipeBarrier<PIPE_V>();
            }
            if (own > 0) {
                Duplicate(rowF32, sumRw, qRowSize);
                PipeBarrier<PIPE_V>();
                Div(acc, acc, rowF32, qRowSize);
                PipeBarrier<PIPE_V>();
            }
            Cast(rowQ, acc, RoundMode::CAST_RINT, qRowSize);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopy(qBarGm_[r * qRowSize], rowQ, qRowSize);

            // w_bar(组内 weights 行同权加权)
            float sumW = 0.0f;
            Duplicate(accW, 0.0f, hSize);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < own; i++) {
                float rw = rwF32.GetValue(i);
                sumW += rw;
                DataCopy(rowQ, callerWeightsGm_[(cumBegin + i) * hSize], hSize);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(rowF32, rowQ, RoundMode::CAST_NONE, hSize);
                PipeBarrier<PIPE_V>();
                Muls(rowF32, rowF32, rw, hSize);
                PipeBarrier<PIPE_V>();
                Add(accW, accW, rowF32, hSize);
                PipeBarrier<PIPE_V>();
            }
            if (own > 0) {
                Duplicate(rowF32, sumW, hSize);
                PipeBarrier<PIPE_V>();
                Div(accW, accW, rowF32, hSize);
                PipeBarrier<PIPE_V>();
            }
            Cast(rowQ, accW, RoundMode::CAST_RINT, hSize);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopy(wBarGm_[r * hSize], rowQ, hSize);
            // WAR 防护(先例 refine_service_vector:806 MTE3_V 同类):上面对 rowQ 的 MTE3 读
            // (qBar/wBar store)与下一轮 DataCopy(rowQ/rwRow, GM→UB) 的 MTE2 复写之间,
            // 必须等 MTE3 读完成;V(Cast 对 rwRow/rwF32 的读)与下轮 MTE2 复写同理。
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        }
        outQueue_.FreeTensor(acc);
    }

    PipeBarrier<PIPE_MTE3>();
    SyncAll();
}

// --------------------------M2.5 窗口注入(AIV 全局后阶段,hasWindow 门控)--------------------------
// 每请求一行:win = [aslk-(g-1), aslk+aslq差分) 与候选行求差集,新增位置按升序 append 到有效前缀后,
// 尾部 -1 补齐到 outW(与 torch _inject_local_window 逐位一致:有效数 = min(upper,c),topk 行内
// -1 恒在尾部);aslk'[r] = validC + nNew。hasWindow=0 时主 pass 已直写输出,仅算 aslk'=min(upper,c)。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessWindow(TPipe *pipe)
{
    // 主 pass 候选写(MTE3)全部完成后才可读回
    PipeBarrier<PIPE_MTE3>();
    SyncAll();
    // 独立窗口缓冲(pipe->Reset 释放主 pass 缓冲,先例 = InitLDBuffers)
    pipe->Reset();
    const uint32_t c = static_cast<uint32_t>(constInfo_.sparseCount);
    const uint32_t W = constInfo_.outW;
    const uint32_t cAlign = IndexerCoarseScreenCommon::Align<uint32_t>(c, 8);
    const uint32_t rowNum = static_cast<uint32_t>(constInfo_.batchSize);
    const uint32_t aivNum = GetBlockNum() * 2;
    const uint32_t rowsPerAiv = IndexerCoarseScreenCommon::CeilDiv(rowNum, aivNum);
    pipe->InitBuffer(winCandBuf_, c * sizeof(int32_t));
    pipe->InitBuffer(winCandF32Buf_, cAlign * sizeof(float));
    pipe->InitBuffer(winPosF32Buf_, cAlign * sizeof(float));
    pipe->InitBuffer(winMskF32Buf_, cAlign * sizeof(float));
    pipe->InitBuffer(winOutBuf_, W * sizeof(int32_t));
    pipe->InitBuffer(winAuxBuf_, (64 + rowsPerAiv) * sizeof(int32_t));

    LocalTensor<int32_t> candI32 = winCandBuf_.Get<int32_t>();
    LocalTensor<float> candF32 = winCandF32Buf_.Get<float>();
    LocalTensor<float> posF32 = winPosF32Buf_.Get<float>();
    LocalTensor<float> mskF32 = winMskF32Buf_.Get<float>();
    LocalTensor<int32_t> outI32 = winOutBuf_.Get<int32_t>();
    LocalTensor<int32_t> auxI32 = winAuxBuf_.Get<int32_t>();

    const uint32_t rBegin = static_cast<uint32_t>(GetBlockIdx()) * rowsPerAiv;
    const uint32_t rEnd = IndexerCoarseScreenCommon::Min(rBegin + rowsPerAiv, rowNum);
    for (uint32_t r = rBegin; r < rEnd; r++) {
        uint32_t upper = aslkGm_.GetValue(r);
        uint32_t cumEnd = callerSeqLenGmQ_.GetValue(r);
        uint32_t cumBegin = (r == 0) ? 0U : callerSeqLenGmQ_.GetValue(r - 1);
        uint32_t own = cumEnd - cumBegin;
        uint32_t validC = IndexerCoarseScreenCommon::Min(upper, c);
        if (!constInfo_.hasWindow) {
            // 纯粗筛模式:主 pass 已直写输出(行距 = outW = coarseCount),仅算 aslk'
            auxI32.SetValue(64 + (r - rBegin), static_cast<int32_t>(validC));
            continue;
        }
        DataCopy(candI32, candidatesWsGm_[r * c], c);
        PipeBarrier<PIPE_MTE2>();
        Cast(candF32, candI32, RoundMode::CAST_NONE, c);
        PipeBarrier<PIPE_V>();

        // 窗口收集:win = [winStart, winEnd) 内的有效位置,不在候选行的按升序 append。
        // validC > 0 时先去重:|cand-pos| 截断到 [0,1] 后树状求和,
        // 和 < validC ⇔ 存在精确相等(present)。位置 ≤ 2^24,fp32 减法精确。
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
                // 两级对齐归并(NPU 实测修复:旧标量树状折叠在 [validC]/[half] 处的
                // 向量操作偏移仅 4B 粒度,触发 507015 "UB address not aligned"):
                // 布局 [64 行,cols 列](cols∈{16,32,64},2 的幂),所有向量操作
                // 偏移 = now*cols*4 ≥ 64B;二级折到 8 元素(偏移 ≥32B)后标量收尾。
                uint32_t nb = IndexerCoarseScreenCommon::CeilDiv(validC, 64U);
                uint32_t cols = (nb <= 16) ? 16 : (nb <= 32) ? 32 : 64;
                // 全区清零再覆写有效段(pad 写避开任意偏移;Sub/Abs/Mins 只写 [0,validC))
                Duplicate(mskF32, 0.0f, 64 * cols);
                PipeBarrier<PIPE_V>();
                Duplicate(posF32, static_cast<float>(pos), validC);
                PipeBarrier<PIPE_V>();
                Sub(mskF32, candF32, posF32, validC);
                PipeBarrier<PIPE_V>();
                Abs(mskF32, mskF32, validC);
                PipeBarrier<PIPE_V>();
                Mins(mskF32, mskF32, 1.0f, validC);
                PipeBarrier<PIPE_V>();
                // 一级:64 行按列折叠(in-place 重叠加法同 DoReduce 先例)
                uint32_t now = 64;
                while (now > 1) {
                    now >>= 1;
                    Add(mskF32, mskF32, mskF32[now * cols], now * cols);
                    PipeBarrier<PIPE_V>();
                }
                // 二级:cols → 8 元素(偏移 n*4 ≥ 32B)
                uint32_t n2 = cols;
                while (n2 > 8) {
                    n2 >>= 1;
                    Add(mskF32, mskF32, mskF32[n2], n2);
                    PipeBarrier<PIPE_V>();
                }
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                float diffSum = 0.0f;
                for (int t = 0; t < 8; t++) {
                    diffSum += mskF32.GetValue(t);
                }
                // aicore 禁止 float↔unsigned 直转,经 int32 中转
                if (diffSum > static_cast<float>(static_cast<int32_t>(validC)) - 0.5f) {
                    continue; // present:窗口位置已在候选行
                }
            }
            auxI32.SetValue(nNew, pos);
            nNew++;
        }

        // 组装输出行:[0,c)=候选行原序(含 -1 尾),[validC, validC+nNew)=窗口新增(升序),[c,W)=-1
        Duplicate(outI32, constInfo_.INVALID_IDX, W);
        PipeBarrier<PIPE_V>();
        DataCopy(outI32, candI32, c);
        PipeBarrier<PIPE_V>();
        if (nNew > 0) {
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            for (uint32_t k = 0; k < nNew; k++) {
                outI32.SetValue(validC + k, auxI32.GetValue(k));
            }
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(candidatesOutGm_[r * W], outI32, {1, static_cast<uint16_t>(W * sizeof(int32_t)), 0, 0});
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
