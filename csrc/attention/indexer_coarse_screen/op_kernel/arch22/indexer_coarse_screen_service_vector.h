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
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> partialTopkGm);
    __aicore__ inline void FlushDeferredMerges(); // M3:SyncAll 后延迟行归并+CopyOut
    __aicore__ inline void ZeroMyMagics(); // M3C:启动前清零本 AIV magic 块(防陈旧命中)
    __aicore__ inline void PublishMyMagic(); // M3C:Flush 入口发布(宣告本核 partial 全落地)
    __aicore__ inline void PollPartialMagic(uint32_t pc, uint32_t seq); // M3C:轮询目标核 line 尾 magic
    __aicore__ inline uint64_t PartialLineBase(uint32_t pairIdx, uint32_t seq) const;
    __aicore__ inline uint64_t PartialMagicBase(uint32_t pairIdx, uint32_t seq) const;
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
    __aicore__ inline void ProcessWindow(TPipe *pipe, uint32_t rBegin, uint32_t rEnd, bool useMask = false);
    // ---- M1a 组均值预阶段(proxy_mode=1;骨架 = b2fe4a23 ProcessGroupMean)----
    __aicore__ inline void InitProxyGlobalTensor(__gm__ uint8_t *rawQuery, __gm__ uint8_t *rawWeights,
                                                 GlobalTensor<Q_T> qBarOut, GlobalTensor<Q_T> wBarOut);
    __aicore__ inline void ProcessGroupMean(uint32_t rBegin, uint32_t rEnd, bool isEmptyRange);

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
    // M1a 预阶段:raw 输入(caller 原样) + q̄/w̄ 写出目标(mode1 = workspace 段)
    GlobalTensor<Q_T> rawQueryGm_;
    GlobalTensor<Q_T> rawWeightsGm_;
    GlobalTensor<Q_T> qBarOutGm_;
    GlobalTensor<Q_T> wBarOutGm_;
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

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int64_t blockS2StartIdx_ = 0;
    GlobalTensor<float> partialTopkGm_; // M3:跨核 fragment partial
    bool fragIsHead_ = false;           // M3:本核 fragment 是否行头片段(s2Idx>0 起)
    // M3:延迟归并记录(fragmented 行每核 ≤2:tail+head 各至多一条)
    uint32_t rowCap_ = 0U; // M3C:槽行维 = batchSize*kHeadNum(每请求一槽,免 tile 间复用)
    bool deferValid_[4] = {false, false, false, false};
    // M3F10:4096 行容量(R=512 门禁 + 生产 prefill 千级 P);越界写踩坏相邻成员致 507015
    uint32_t winRowMask_[128] = {0U}; // 本核写过的行(完整行+fold行),窗口读者=写者
    uint32_t deferCStart_[4] = {0U, 0U, 0U, 0U};
    uint32_t deferS1Begin_[4] = {0U, 0U, 0U, 0U};
    uint32_t deferS1Num_[4] = {0U, 0U, 0U, 0U};
    uint64_t deferIndiceOff_[4] = {0U, 0U, 0U, 0U};
    uint32_t deferSeq_[4] = {0U, 0U, 0U, 0U};
    uint32_t deferFoldFrom_[4] = {0U, 0U, 0U, 0U}; // M3C:head 槽折叠起点 = cstart+1
    uint32_t deferRowBase_[4] = {0U, 0U, 0U, 0U}; // M3C:槽行基 = indiceOutOffset/sparseCount
    // pair-merge v2: 延迟奇数块存 reduceOutBuf_ 保留段 [2V,4V)(零新增 UB),
    // 偶数块 MrgSort(A,B)→tmp 后一次归并进 acc(top-k 关联律, 输出不变)
    bool pairFilled_[8] = {false};
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

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
    rowCap_ = static_cast<uint32_t>(constInfo.batchSize * constInfo.kHeadNum); // M3C
}

template <typename LIT>
__aicore__ inline void
IndexerCoarseScreenServiceVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                        GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                        GlobalTensor<float> partialTopkGm)
{
    this->mm1ResGm = mm1ResGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->partialTopkGm_ = partialTopkGm;
}

// M3:SyncAll 后的延迟行归并(行尾 fragment 所在核 = 属主)。
// 槽 = (pairIdx*2 + seq);本核 tail 片段槽 seq=fragIsHead_?1:0,其余核:
// cstart=行首所在核 → seq 0,其余(行头片段)→ 1。
// M3C:槽几何 —— line = (pairIdx*2+seq);每 line = rowCap_ 行 × 8192 fp32 + 16 magic 尾
template <typename LIT>
__aicore__ inline uint64_t IndexerCoarseScreenServiceVector<LIT>::PartialLineBase(
    uint32_t pairIdx, uint32_t seq) const
{
    return (static_cast<uint64_t>(pairIdx) * 2U + seq) * (static_cast<uint64_t>(rowCap_) * 8192U + 16U);
}

template <typename LIT>
__aicore__ inline uint64_t IndexerCoarseScreenServiceVector<LIT>::PartialMagicBase(
    uint32_t pairIdx, uint32_t seq) const
{
    return PartialLineBase(pairIdx, seq) + static_cast<uint64_t>(rowCap_) * 8192U;
}

// M3C:启动前清零本 AIV 的 magic 32B 块(seq=1 line 尾) —— 防 launch 间陈旧命中
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ZeroMyMagics()
{
    LocalTensor<float> z = outQueue_.AllocTensor<float>();
    Duplicate(z, 0.0f, 8U);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    // M3D:seq=0/1 两条 line 都要清 —— 属主会 poll 行首核的 seq=0 magic
    for (uint32_t seq = 0; seq < 2U; seq++) {
        uint64_t base = PartialMagicBase(static_cast<uint32_t>(blockId_ >> 1), seq)
                      + ((blockId_ & 1U) ? 8U : 0U);
        DataCopy(partialTopkGm_[base], z, 8U);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(z);
    PipeBarrier<PIPE_ALL>();
}

// M3C:发布本 AIV magic —— 本核全部 partial 写完后(Flush 入口)调用,
// 同核 MTE3 FIFO 保证数据先于 magic 落地
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::PublishMyMagic()
{
    constexpr uint32_t MAGIC = 0x5A5A00FFU;
    LocalTensor<float> m = outQueue_.AllocTensor<float>();
    Duplicate(m.ReinterpretCast<int32_t>(), static_cast<int32_t>(MAGIC), 8U);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    for (uint32_t seq = 0; seq < 2U; seq++) { // M3D:两条 line 都发布
        uint64_t base = PartialMagicBase(static_cast<uint32_t>(blockId_ >> 1), seq)
                      + ((blockId_ & 1U) ? 8U : 0U);
        DataCopy(partialTopkGm_[base], m, 8U);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(m);
    PipeBarrier<PIPE_ALL>();
}

// M3C:spin-poll 目标核 line 尾两个 magic 块(偶+奇 AIV,64B 一读);带上限 fail-open
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::PollPartialMagic(uint32_t pc, uint32_t seq)
{
    constexpr uint32_t MAGIC = 0x5A5A00FFU;
    uint64_t base = PartialMagicBase(pc, seq);
    bool ready = false;
    for (uint32_t spin = 0; spin < 8000000U && !ready; spin++) {
        // M3W:事件旗标是单信用信号量,不可作自旋旗(第二轮 WaitFlag 永等挂死);
        // PIPE_ALL 屏障无信用语义,作 tmpUb_ 复用护栏
        PipeBarrier<PIPE_ALL>();
        DataCopy(tmpUb_, partialTopkGm_[base], 16U);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        ready = tmpUb_.ReinterpretCast<int32_t>().GetValue(0) == static_cast<int32_t>(MAGIC)
             && tmpUb_.ReinterpretCast<int32_t>().GetValue(8) == static_cast<int32_t>(MAGIC);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::FlushDeferredMerges()
{
    // M3F2 折叠 v2:入口发布自身 magic(宣告本核 partial 全落地),等齐贡献核后折叠。
    // 原语 = FoldMerge4(4×2048 单次归并,段长合法);槽间 PIPE_ALL(M3Z3)。
    // defer 槽 0..3 全遍历;贡献槽 = 行首核 seq0 + (行首核,本核) 区间各核 seq1。
    // M3F2C:同核 MTE3→MTE2 事件同步 —— 本核槽写入(主 pass MTE3)必须先于
    // 本核槽读取(Flush MTE2)落地;PipeBarrier 不等跨管道完成(2026-09-17 同款)。
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    // M3F4:全核屏障 —— 消灭 magic 陈旧竞态(poller 在慢核清零前读到 N-1 残留
    // magic → 过早折叠撕裂槽)。屏障后全部主 pass(含各自 ZeroMyMagics)已完成,
    // PublishMyMagic 后任何 poll 必见本 launch magic。参与集 = 全部 AIV(同窗口)。
    SyncAll();
    PublishMyMagic();
    for (uint32_t r = 0; r < 4U; r++) {
        if (!deferValid_[r]) {
            continue;
        }
        int64_t offset = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? virTopK : constInfo_.sparseCount / 2;
        int64_t copyLen = (constInfo_.sparseCount <= SPARSE_COUNT_4K)
                        ? constInfo_.sparseCount
                        : constInfo_.sparseCount / 2;
        int64_t copyNum = (constInfo_.sparseCount <= SPARSE_COUNT_4K) ? 1 : 2;
        // M3C:先等齐 —— 行首核 tail 槽(seq=0)+ head 贡献核(seq=1),再折叠
        PollPartialMagic(deferCStart_[r], 0U);
        for (uint32_t pc = deferFoldFrom_[r]; pc < static_cast<uint32_t>(blockId_ >> 1); pc++) {
            PollPartialMagic(pc, 1U);
        }
        for (uint32_t innerS1Idx = 0; innerS1Idx < deferS1Num_[r]; innerS1Idx++) {
            // M3F3:分裂行 ws 写者 = 本核(fold) → 记入窗口行位图
            uint32_t wrD = deferRowBase_[r] + innerS1Idx * static_cast<uint32_t>(constInfo_.kHeadNum);
            if (wrD < 4096U) { winRowMask_[wrD >> 5] |= (1U << (wrD & 31U)); }
            // 复位运行 topk 区,装本核行槽
            InitSortOutBuf(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK * 2);
            // M3F7:V 预填排空后才可 MTE2 载入覆写 acc(窗口组装路径同款事件对;
            // 缺此对 = MTE2 与 -1 预填同址竞态 → 部分 -1 行逐 launch 漂移)
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            PipeBarrier<PIPE_V>();
            uint64_t slotRow = static_cast<uint64_t>(deferRowBase_[r])
                             + static_cast<uint64_t>(innerS1Idx) * constInfo_.kHeadNum;
            uint64_t own = PartialLineBase(static_cast<uint32_t>(blockId_ >> 1), deferSeq_[r])
                         + slotRow * 8192U;
            DataCopy(globalTopkUb_[innerS1Idx * virTopK * 2],
                     partialTopkGm_[own], 4096U * 2U);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            PipeBarrier<PIPE_V>();
            for (uint32_t pc = deferCStart_[r]; pc < static_cast<uint32_t>(blockId_ >> 1); pc++) {
                uint32_t pseq = (pc == deferCStart_[r]) ? 0U : 1U;
                uint64_t pslot = PartialLineBase(pc, pseq) + slotRow * 8192U;
                LocalTensor<float> pbuf = outQueue_.AllocTensor<float>();
                PipeBarrier<PIPE_ALL>(); // M3Z3:上一轮归并/Extract 落定后再启 MTE2
                DataCopy(pbuf, partialTopkGm_[pslot], 4096U * 2U);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                PipeBarrier<PIPE_V>();
                IndexerCoarseScreenServiceVec::FoldMerge4(globalTopkUb_[innerS1Idx * virTopK * 2],
                                                          pbuf, tmpUb_);
                PipeBarrier<PIPE_ALL>();
                outQueue_.FreeTensor(pbuf);
            }
            for (int64_t i = 0; i < copyNum; i++) {
                LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                Extract(outValueUb, outIdxUb,
                        globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset], offset / 32);
                LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                outQueue_.EnQue<float>(outValueUb);
                outValueUb = outQueue_.DeQue<float>();
                IndexerCoarseScreenServiceVec::CopyOut(
                    indiceOutGm[deferIndiceOff_[r] + (deferS1Begin_[r] + innerS1Idx) *
                                constInfo_.sparseCount + i * offset],
                    idxULocal1, copyLen);
                outQueue_.FreeTensor(outValueUb);
            }
        }
        deferValid_[r] = false;
    }
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
    int32_t cuS1ProcHalf = CeilDiv(cuS1ProcNum, 2); // P3: 一行两用,免每块 2 次整除
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? cuS1ProcHalf : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * cuS1ProcHalf;

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * cuS1ProcHalf * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * cuS1ProcHalf * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
        for (int i = 0; i < 8; i++) { pairFilled_[i] = false; }
        fragIsHead_ = false; // M3:行首在本核 → tail/whole 片段
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
        for (int i = 0; i < 8; i++) { pairFilled_[i] = false; }
        fragIsHead_ = info.s2Idx != 0; // M3:核范围起于行中 → head 片段
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq(粗筛域上界 upper[r],PA 布局恒绝对值;
    // attenMaskFlag 硬编码 false,掩码改写分支已随死代码删除)
    int32_t cuRealAcSeq = info.actS2Size;
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
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
            // over-2K(host 拒绝 coarseCount≤2048)恒走 SortAll + 原生单次 MergeSort
            // (virTopK)= 原生 lightning_indexer 生产同款。actS1Size≡1(TND 每请求单
            // 代理行),actS1Size>4 缓存路由与非 over-2K 精排路径不可达,已删;历史上
            // v11 双累积形态从未 NPU 验证且多 chunk 实测 fault,2026-09-16 废弃。
            SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            // pair-merge v2: 奇数块延迟入保留段 [2V,4V); 偶数块 MrgSort(A,B)→tmp
            // 后一次归并进 acc —— 每对省一次 4096 宽归并 + 32KB 拷回
            if (!pairFilled_[innerS1Idx]) {
                // 双向事件(ID2 此点空闲): 源 [0,2V) 为 V 产出须先落定;
                // 拷回(MTE2)完成才可被 V(MrgSort4) 读 —— 缺此对即 VEC 竞态
                AscendC::SetFlag<HardEvent::V_MTE2>(2);
                AscendC::WaitFlag<HardEvent::V_MTE2>(2);
                AscendC::DataCopy(reduceOutBuff[s2BaseSize_ * 2], reduceOutBuff,
                                  cuS2LenVecAlign * 2);
                AscendC::SetFlag<HardEvent::MTE2_V>(2);
                AscendC::WaitFlag<HardEvent::MTE2_V>(2);
                PipeBarrier<PIPE_V>();
                pairFilled_[innerS1Idx] = true;
            } else {
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = cuS2LenVecAlign;
                params.elementLengths[1] = s2BaseSize_;
                params.ifExhaustedSuspension = false;
                params.validBit = 0b0011;
                params.repeatTimes = 1;
                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = reduceOutBuff;
                srcList.src2 = reduceOutBuff[s2BaseSize_ * 2];
                AscendC::MrgSort<float>(tmpSortBuf, srcList, params);
                PipeBarrier<PIPE_V>();
                IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK,
                                        tmpSortBuf, s2BaseSize_ * 2, tmpUb_);
                pairFilled_[innerS1Idx] = false;
            }
            SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool fragEnd = info.isLastS2InnerLoop;
            if (fragEnd && pairFilled_[innerS1Idx]) {
                LocalTensor<float> deferred = reduceOutBuff[s2BaseSize_ * 2];
                IndexerCoarseScreenServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK,
                                        deferred, s2BaseSize_, tmpUb_);
                pairFilled_[innerS1Idx] = false;
            }
            if (fragEnd) {
                // M3:fragment 收尾 —— 运行 top-4096 写本 pair 槽(head=1/tail=0);
                // 行尾且为 head 片段 → 跨核 fragmented 行,记录延迟归并(SyncAll 后)
                // M3C:槽 per-请求行(line = pair*2+seq;行 = indiceOutOffset/sparseCount),
                // 消除 tile(请求)间槽复用 → 一次 Flush 折叠即正确
                // M3F5B:槽写(MTE3 读 acc)前等 V 合并落定 ——
                // V_MTE3 = 窗口循环尾同款生产事件对;F2D 失败疑因 MTE2_MTE3 陌生对
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                PipeBarrier<PIPE_ALL>();
                uint32_t seq = fragIsHead_ ? 1U : 0U;
                uint64_t wslotRow = static_cast<uint64_t>(info.indiceOutOffset / constInfo_.sparseCount)
                                  + static_cast<uint64_t>(innerS1Idx) * constInfo_.kHeadNum;
                uint64_t pslot = PartialLineBase(static_cast<uint32_t>(blockId_ >> 1), seq)
                               + wslotRow * 8192U;
                // M3F6R:直写恢复(F7 读侧 V_MTE2 为真根因修复;队列协议系误诊产物)
                DataCopy(partialTopkGm_[pslot],
                         globalTopkUb_[innerS1Idx * virTopK * 2], 4096U * 2U);
                // M3F5:槽写(MTE3 读 acc)排空后下一 tile 才可覆写 acc
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                if (isS2End && fragIsHead_) { // M3C:按行记录(每 tile 恒 1 行)
                    // cstart 反向步进:unitStart(c)=c*cap+min(c,deal1More) ≤ uStart
                    uint32_t pairIdx = static_cast<uint32_t>(blockId_ >> 1);
                    uint64_t uEnd = static_cast<uint64_t>(info.unitStartSelf) +
                        static_cast<uint64_t>(info.s2Idx - static_cast<uint32_t>(blockS2StartIdx_) + 1U);
                    uint32_t unitsInRow = (info.actS2Size + s2BaseSize_ - 1) / s2BaseSize_;
                    uint64_t uStart = uEnd - unitsInRow;
                    uint32_t cstart = pairIdx;
                    while (cstart > 0) {
                        uint32_t pc = cstart - 1;
                        uint64_t us = static_cast<uint64_t>(pc) * info.unitCap +
                            (pc < info.deal1More ? pc : info.deal1More);
                        cstart = pc; // M3I:pc 恒入候选;us<=uStart 即 pc=行首核
                        if (us <= uStart) { break; }
                    }
                    uint32_t slot = 4U;
                    for (uint32_t e = 0; e < 4U; e++) {
                        if (!deferValid_[e]) { slot = e; break; }
                    }
                    if (slot < 4U) { // M3C:容量 4(R≤96);溢出行退化为单核结果
                        deferValid_[slot] = true;
                        deferCStart_[slot] = cstart;
                        deferFoldFrom_[slot] = cstart + 1U; // 行首核走 seq=0 槽,head 槽从下一核起
                        deferRowBase_[slot] = static_cast<uint32_t>(
                            info.indiceOutOffset / constInfo_.sparseCount);
                        deferS1Begin_[slot] = static_cast<uint32_t>(cuS1BeginIdxPerAiv);
                        deferS1Num_[slot] = static_cast<uint32_t>(cuS1ProcNumPerAiv);
                        deferIndiceOff_[slot] = static_cast<uint64_t>(info.indiceOutOffset);
                        deferSeq_[slot] = seq;
                    }
                }
            }
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End && !fragIsHead_;
            if (needCopyOutGm) {
                // M3F3:完整行 ws 写者 = 本核 → 记入窗口行位图
                uint32_t wr = static_cast<uint32_t>(info.indiceOutOffset / constInfo_.sparseCount)
                            + static_cast<uint32_t>(innerS1Idx) * constInfo_.kHeadNum;
                if (wr < 4096U) { winRowMask_[wr >> 5] |= (1U << (wr & 31U)); }
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


// --------------------------M1a 组均值代理(AIV 全局预阶段,proxyMode=1)--------------------------
// q_bar[r,h,:] = (Σ_{i<own_r} query[cum_{r-1}+i,h,:]) / own_r —— fp32 顺序累加 +
// Div + 一次 CAST_RINT bf16,与 torch view(K,g,H,*).mean(dim=1) 同序同型
// (probe_m1a A/B 逐位实证)。骨架照抄 b2fe4a23 ProcessGroupMean(NPU 验证版),
// 三处简化:无 row_weights(均匀均值)、无 proxyCum(现行 prefix 硬编码 bIdx)、
// 写目标 = workspace qBarWs/wBarWs(Init 按 mode 绑定)。
// 行分区:全部 AIV 均分 R 行(全员一致进入/到达屏障);own==0 行跳过(无 mm1
// 消费,DealActSeqLenIsZero 行安全)。收尾 PipeBarrier<MTE3>(本核 flush)+
// SyncAll(全 AIV 屏障)→ 所有 AIV 的 q̄/w̄ 写全局可见;ProcessMain 随后预置
// syncV1C1×2(MTE3 管道),AIC 首个 matmul 的 CrossCoreWaitFlag 由此放行。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::InitProxyGlobalTensor(
    __gm__ uint8_t *rawQuery, __gm__ uint8_t *rawWeights,
    GlobalTensor<Q_T> qBarOut, GlobalTensor<Q_T> wBarOut)
{
    rawQueryGm_.SetGlobalBuffer((__gm__ Q_T *)rawQuery);
    rawWeightsGm_.SetGlobalBuffer((__gm__ Q_T *)rawWeights);
    qBarOutGm_ = qBarOut;
    wBarOutGm_ = wBarOut;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessGroupMean(uint32_t rBegin, uint32_t rEnd,
                                                                               bool isEmptyRange)
{
    // 行分区 = 本 pair AIC 的 SplitCore 精确请求区间(main pass 消费域);行由
    // 本 pair 偶 AIV 独占计算(见下行单写者注记)。空区间(无块 pair)零写入零同步。
    if (!constInfo_.proxyMode || isEmptyRange) {
        return; // mode 0:q̄/w̄ caller 直绑;空 pair:无消费无写出
    }
    // 行级单写者配对(2026-10-09 NPU 定案):奇 AIV 跳过预阶段 —— 行 p 由
    // pair p 的偶 AIV 独占计算与写出。双 AIV 冗余同址 MTE3 store 会并发互踩
    // (词级撕裂,候选值混入 raw 行/0);单写者同时把预阶段工作量减半。
    if (GetBlockIdx() % 2 == 1) {
        return;
    }

    // pair 内两 AIV 重复计算同一区间(写值相同,并发同值写无撕裂):配合
    // per-pair flag id,本对 AIC 消费的每条贷记都由本对 AIV 在全部行写毕后
    // 置位 —— 无早读。twin 同址写在不同构建实测稳定(撕裂根因是全局贷记)。
    const uint32_t qRowSize = static_cast<uint32_t>(constInfo_.headDim * constInfo_.gSize); // g*Dh(q̄ 行宽;kernel qHeadNum≡gSize)
    const uint32_t hSize = static_cast<uint32_t>(constInfo_.gSize); // g(w̄ 行宽)
    // rEnd 已为半开(调用点 +1)
    const uint32_t rEndEff = IndexerCoarseScreenCommon::Min(rEnd,
        static_cast<uint32_t>(constInfo_.batchSize));
    if (rBegin < rEnd) {
        // UB 借道(运行于主 pass 首个 ProcessVec 之前,缓冲皆空闲;不触 indexBuf_/
        // sortOutBuf_ —— InitBuffers 的 ArithProgression/InitSortOutBuf 状态必须存活)。
        // tmpBuf_ 共 20KB:rowQ 双缓冲 bf16 [0,8KB)/[8KB,16KB),rowF32 槽深
        // CHUNK=1024 float 置 tmpUb_[qRowSize](字节 [16KB,20KB))—— 恰满不越界。
        // 2026-10-09 定案:旧 rowF32 = tmpUb_[2*qRowSize](字节 [32KB,48KB))越界
        // 12KB 踩毁 sortOutBuf_,是预阶段确定性全错的根因(非同步竞态)。
        constexpr uint32_t CHUNK = 1024U; // rowF32 槽深(float);q̄ 行按此分段
        LocalTensor<Q_T> rowQ0 = tmpUb_.template ReinterpretCast<Q_T>();
        LocalTensor<Q_T> rowQ1 = tmpUb_.template ReinterpretCast<Q_T>()[qRowSize];
        LocalTensor<float> rowF32 = tmpUb_[qRowSize];
        LocalTensor<float> accW = brcBuf_.Get<float>();
        uint32_t prevCum = (rBegin == 0) ? 0U : static_cast<uint32_t>(callerSeqLenGmQ_.GetValue(rBegin - 1));
        for (uint32_t r = rBegin; r < rEnd; r++) {
            uint32_t cumEnd = static_cast<uint32_t>(callerSeqLenGmQ_.GetValue(r));
            uint32_t own = cumEnd - prevCum; // P3: cumBegin≡上行 cumEnd,滚动缓存
            prevCum = cumEnd;
            if (own == 0) {
                continue; // 空请求行:无 mm1 消费,q̄/w̄ 不写
            }
            uint32_t cumBegin = cumEnd - own;
            LocalTensor<float> acc = outQueue_.AllocTensor<float>();
            // q̄:fp32 顺序累加 own 行 → Div(own) → 一次 bf16 舍入(torch mean 同序)。
            // [fix5 基线回滚 2026-10-09]队列化重构(TQue 载入 + outQueue_ store)
            // 与 PipeBarrier 化两版均确定性全错(probe 计数恒等),唯一 ALL-PASS
            // 基线 = SetWaitFlag 全套护栏 + rowQ0 直 store;性能优化须以此为
            // 基线逐项重做并单独验证(SetWaitFlag ~130µs/次待解)。
            Duplicate(acc, 0.0f, qRowSize);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < own; i++) {
                // RAW:copy 后 MTE2 排空才放行 V 读;WAR:i≥2 复用同 buffer,
                // 复写前等上上轮 V 读完(own≥3 才暴露)
                if (i >= 2) {
                    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                }
                LocalTensor<Q_T> rowQ = (i % 2 == 0) ? rowQ0 : rowQ1;
                DataCopy(rowQ, rawQueryGm_[(cumBegin + i) * qRowSize], qRowSize);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                // 逐元素 Cast/Add 按 CHUNK 分段(rowF32 槽深限制),不改位型
                for (uint32_t c = 0; c < qRowSize / CHUNK; c++) {
                    Cast(rowF32, rowQ[c * CHUNK], RoundMode::CAST_NONE, CHUNK);
                    PipeBarrier<PIPE_V>();
                    Add(acc[c * CHUNK], acc[c * CHUNK], rowF32, CHUNK);
                    PipeBarrier<PIPE_V>();
                }
            }
            for (uint32_t c = 0; c < qRowSize / CHUNK; c++) {
                Duplicate(rowF32, static_cast<float>(static_cast<int32_t>(own)), CHUNK);
                PipeBarrier<PIPE_V>();
                Div(acc[c * CHUNK], acc[c * CHUNK], rowF32, CHUNK);
                PipeBarrier<PIPE_V>();
            }
            Cast(rowQ0, acc, RoundMode::CAST_RINT, qRowSize);
            PipeBarrier<PIPE_V>();
            // V 写 rowQ0 → MTE3 读 rowQ0:V→MTE3 事件保序(缺此 store 读到 stale)
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopy(qBarOutGm_[r * qRowSize], rowQ0, qRowSize);
            // q̄ store(MTE3 读 rowQ0)与 w̄ 首轮 DataCopy(MTE2 写 rowQ0)跨段 WAR
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            // w̄:同一组行对 weights [N,H] 同式
            Duplicate(accW, 0.0f, hSize);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < own; i++) {
                // 同 q̄ 循环:i≥2 复用同 buffer,复写前等上上轮 V 读完
                if (i >= 2) {
                    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                }
                LocalTensor<Q_T> rowQ = (i % 2 == 0) ? rowQ0 : rowQ1;
                // w̄ 有效宽 hSize×2B=8B(<32B 下限):借真实行距 H×2B=64B,读 32
                // 元素(64B)只用前 hSize 个 —— 界内有保证(raw w 尾行 8(N-1)+64 ≤ 64N)
                DataCopy(rowQ, rawWeightsGm_[(cumBegin + i) * hSize], 32);
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                Cast(rowF32, rowQ, RoundMode::CAST_NONE, hSize);
                PipeBarrier<PIPE_V>();
                Add(accW, accW, rowF32, hSize);
                PipeBarrier<PIPE_V>();
            }
            Duplicate(rowF32, static_cast<float>(static_cast<int32_t>(own)), hSize);
            PipeBarrier<PIPE_V>();
            Div(accW, accW, rowF32, hSize);
            PipeBarrier<PIPE_V>();
            Cast(rowQ0, accW, RoundMode::CAST_RINT, hSize);
            PipeBarrier<PIPE_V>();
            // 同上:V 写 rowQ0 → MTE3 读,先排空 V
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            // 8B store 走 DataCopyPad
            DataCopyPad(wBarOutGm_[r * hSize], rowQ0,
                        {1, static_cast<uint16_t>(hSize * sizeof(Q_T)), 0, 0});
            // WAR 防护:本行对 rowQ 的 MTE3 读与下一行 MTE2 复写、V 读与下轮
            // MTE2 复写之间必须全序
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            outQueue_.FreeTensor(acc);
        }
    }
    // 收尾仅本核 MTE3 排空;跨核可见性由 per-pair 种子 flag 背书(本核全部
    // GM 写先于置位,同核同管道程序序)
    PipeBarrier<PIPE_MTE3>();
}


// --------------------------M2.5 窗口注入(AIV 全局后阶段,hasWindow 门控)--------------------------
// 每请求一行:win = [aslk-(g-1), aslk+aslq差分) 与候选行求差集,新增位置按升序 append 到有效前缀后,
// 尾部 -1 补齐到 outW(与 torch _inject_local_window 逐位一致:有效数 = min(upper,c),topk 行内
// -1 恒在尾部);aslk'[r] = validC + nNew。hasWindow=0 时主 pass 已直写输出,仅算 aslk'=min(upper,c)。
template <typename LIT>
__aicore__ inline void IndexerCoarseScreenServiceVector<LIT>::ProcessWindow(TPipe *pipe, uint32_t rBegin, uint32_t rEnd, bool useMask)
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
    // 2026-09-23 竞态第四例(graph_smoke_service_shape.py S7 二分判决):
    //   批内含 aslk=0 行(零行走 DealActSeqLenIsZero 清理分支)时,实数行
    //   候选集非确定损坏(C_diff 2507~4043 随时序漂移,同输入紧邻两跑互差
    //   998;hasWindow=0 纯粗筛零污染 = 窗口阶段定点)。曾试本入口加
    // 2026-09-24 P6:入口三件套(PIPE_ALL + MTE3_MTE2 + SyncAll)移除。历史:为
    //   "竞态第四例"所加的深度防御,S7 实验已证其对该竞态无效(真因 = SplitCore
    //   分核 bug,2026-09-24 已根修);配对分区设计下窗口读者=同对写者,跨对无
    //   可见性依赖,SyncAll 纯属每 launch 固定开销。全零批路径(KERN Process
    //   usedCoreNum==0 分支)的"全部 AIV 到屏障"约束随屏障移除一并消失(各 AIV
    //   独立进入、独立 Reset 自身 pipe,无跨核对依赖)。
    // 2026-09-28 回滚1(d7002bac 部分恢复): 入口三件套恢复——生产 burst/gsm8k
    //   在 P6 移除该屏障后随机 AICORE fault(VEC 参数非法, IndexerRefine AIV 尾部),
    //   10:27 SVEC 窗口行循环优化改变了循环结构, 12:46 的"对该竞态无效"结论不再
    //   覆盖新形态; 恢复为深度防御。
    AscendC::PipeBarrier<PIPE_ALL>();
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    SyncAll();
    // 独立窗口缓冲(pipe->Reset 释放主 pass 缓冲,先例 = InitLDBuffers)
    pipe->Reset();
    const uint32_t c = static_cast<uint32_t>(constInfo_.sparseCount);
    const uint32_t W = constInfo_.outW;
    const uint32_t cAlign = IndexerCoarseScreenCommon::Align<uint32_t>(c, 8);
    const uint32_t rowNum = static_cast<uint32_t>(constInfo_.batchSize);
    // 2026-09-24 对齐布局:全部窗口缓冲向上取整到 32B 倍数(int32 即 8 元素)。
    // winOutBuf_ 原尺寸 W*4=16412 非 32B 倍数,P1 双读合并(dedup 零偏移 VEC 读
    // outI32)在其上触发 0x800 UB 非对齐硬件异常;后续缓冲基址亦被带偏。
    pipe->InitBuffer(winCandBuf_, IndexerCoarseScreenCommon::Align<uint32_t>(c, 8) * sizeof(int32_t));
    pipe->InitBuffer(winPosBuf_, cAlign * sizeof(int32_t));
    pipe->InitBuffer(winMskBuf_, cAlign * sizeof(int32_t));
    pipe->InitBuffer(winOutBuf_, IndexerCoarseScreenCommon::Align<uint32_t>(W, 8) * sizeof(int32_t));
    pipe->InitBuffer(winAuxBuf_,
                     IndexerCoarseScreenCommon::Align<uint32_t>(64 + (rEnd - rBegin), 8) * sizeof(int32_t));

    // winCandBuf_:dedup 装载缓冲。2026-09-24 P1 双读合并(outI32 复用)尝试在
    // NPU 上触发 VEC UB 非对齐硬件异常(0x800,winOutBuf_ 尺寸 W*4=16412 非 32B
    // 倍数,Sub 零偏移读踩雷),已回退;双读合并需先解决 TPipe 缓冲对齐布局。
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
        uint32_t prevCum = (rBegin == 0) ? 0U : callerSeqLenGmQ_.GetValue(rBegin - 1);
        for (uint32_t r = rBegin; r < rEnd; r++) {
            bool rowWritten = (r < 4096U) && (winRowMask_[r >> 5] & (1U << (r & 31U))); // 外审6.3:读侧同守卫
            if (useMask && !rowWritten && aslkGm_.GetValue(r) != 0U) { continue; }
            if (useMask) { prevCum = (r == 0) ? 0U : callerSeqLenGmQ_.GetValue(r - 1); }
            DataCopy(dumpCand, candidatesWsGm_[r * c], c);
            PipeBarrier<PIPE_MTE2>();
            SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            uint32_t up = aslkGm_.GetValue(r);
            uint32_t validC = IndexerCoarseScreenCommon::Min(up, c);
            uint32_t cumEnd = callerSeqLenGmQ_.GetValue(r);
            uint32_t own = cumEnd - prevCum; // P3: cumBegin≡上行 cumEnd
            prevCum = cumEnd;
            int32_t winStart = static_cast<int32_t>(up) - static_cast<int32_t>(constInfo_.groupSize - 1);
            int32_t winEnd = static_cast<int32_t>(up) + static_cast<int32_t>(own);
            for (uint32_t j = 0; j < 7; j++) {
                // dumpCand 只装载了 c 个元素;up-3+j 可能 >= c(长前缀行)或下溢
                // (pad 行 up=0)——域外/下溢统一写 -12345 标记,禁止裸标量越界
                // (hasWindow=2 dump 在 up=81920 用例 AIV scalar-UB OOB 崩,
                // _mix_aiv+0x74c8,2026-09-23)。
                uint32_t idx = up - 3U + j;
                outI32.SetValue(16 + j,
                    (up >= 3U + j && idx < validC) ? dumpCand.GetValue(idx)
                                                   : static_cast<int32_t>(-12345));
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
            // v2 指纹词24..27,判"未写 / 写错 / 写串行"三类根因(S8 证据:
            // r1 up=300<c 在窗口入口缺数学上必在的候选 → ws 在窗口前已坏):
            //   24 = Σcand[0..validC) 标量累加(debug 路径,4096 次标量读可接受)
            //   25 = cand[0]   26 = cand[validC-1]   27 = cand[up-1](guarded)
            // 与 hw=0 纯粗筛 eager 参考行对拍:指纹==0xa5 毒化 → 主 pass 未写;
            // 指纹==其他行参考 → 行基址错位(写串行);其余不符 → 写错内容。
            int32_t rowSum = 0;
            for (uint32_t t = 0; t < validC; t++) {
                rowSum += dumpCand.GetValue(t);
            }
            outI32.SetValue(24, rowSum);
            outI32.SetValue(25,
                (validC > 0) ? dumpCand.GetValue(0) : static_cast<int32_t>(-12345));
            outI32.SetValue(26,
                (validC > 0) ? dumpCand.GetValue(validC - 1U) : static_cast<int32_t>(-12345));
            outI32.SetValue(27,
                (up >= 1U && up <= c) ? dumpCand.GetValue(up - 1U)
                                      : static_cast<int32_t>(-12345));
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(candidatesOutGm_[r * W], outI32, {1, static_cast<uint16_t>(28 * sizeof(int32_t)), 0, 0});
            auxI32.SetValue(0, static_cast<int32_t>(aslkGm_.GetValue(r)));
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(aslkOutGm_[r], auxI32, {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
            // 2026-09-24 同款跨行竞态(dump 分支):行间 outI32 由 S 管道 SetValue
            // 复用,上一行 DataCopyPad(MTE3)未落地即被覆写。MTE3_S 事件对
            // (生产 A3 先例 lightning_indexer_service_vector:465)。
            SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        return;
    }
    uint32_t prevCum = (rBegin == 0) ? 0U : callerSeqLenGmQ_.GetValue(rBegin - 1);
    for (uint32_t r = rBegin; r < rEnd; r++) {
        uint32_t upper = aslkGm_.GetValue(r);
        // M3F9:无写者行仅零域行(upper==0)进入 dedup —— validC==0 恒等分支
        // (全零批同款),偶 AIV 冗余写同内容竞态无害;非零域行必有唯一写者核
        bool rowWritten = (r < 4096U) && (winRowMask_[r >> 5] & (1U << (r & 31U))); // 外审6.3:读侧同守卫
        if (useMask && !rowWritten && upper != 0U) { continue; }
        if (useMask) { prevCum = (r == 0) ? 0U : callerSeqLenGmQ_.GetValue(r - 1); }
        uint32_t cumEnd = callerSeqLenGmQ_.GetValue(r);
        uint32_t own = cumEnd - prevCum; // P3: cumBegin≡上行 cumEnd,滚动缓存省每行 1 次标量 GM 读
        prevCum = cumEnd;
        uint32_t validC = IndexerCoarseScreenCommon::Min(upper, c);
        // 2026-09-21 补:validC==0 行的 MTE3(V Identity 写)与本行/下一行
        // MTE2 装载的跨管道序,由下方分支内 MTE2_V/V_S 的既有配对保证;
        // 此处不再有其他裸写。
        if (!constInfo_.hasWindow) {
            // 纯粗筛模式:主 pass 已直写输出(行距 = outW = coarseCount),仅算 aslk'
            if (useMask) {
                auxI32.SetValue(64, static_cast<int32_t>(validC));
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(aslkOutGm_[r], auxI32[64], {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
            } else {
                auxI32.SetValue(64 + (r - rBegin), static_cast<int32_t>(validC));
            }
            continue;
        }
        // 2026-09-17 竞态终修(A3 判决实证):dedup 首个 Sub(V) 可越过未完成的
        // MTE2 拷贝读候选行 —— winStart 位置被判"缺席"追加重复(A3 的 61 即证)。
        // PipeBarrier<PIPE_MTE2> 只排 MTE2 内部;跨管道 MTE2→V 必须事件同步
        // (M1/复刻链同款先例:复刻链因先做 MTE2_S 等待而全对,真实链缺此事件)。
        // validC==0(粗筛域空,如 prefill 首组)无去重,candI32 无人消费;且全零批
        // 的 workspace 为 0 尺寸,读之挂起 —— 装载与等待一并跳过(2026-09-21)。
        if (validC > 0) {
            DataCopy(outI32, candidatesWsGm_[r * c], c); // P1: 装载即组装缓冲,免第二次 16KB GM 读
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        }

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
                Sub(mskI32, outI32, posI32, validC); // P1: 候选行 = outI32[0,c)
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
                // 二级:cols → 8 元素(偏移 n*4 ≥ 32B)。P2 的"折叠到 1"曾引入
                // 16/8/4B 偏移的 Add —— NPU 实测 0x800 VEC 非对齐异常嫌疑,
                // 回退 8 元素形态(2026-09-24);标量侧 8 次 GetValue 保留。
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
        // 2026-09-23 竞态第五例修复(S10 内容判决):坏行 = -1 预填透出
        // (PARTIAL,-1 计数=C_diff-1,有效前缀边界随 launch 漂移)。旧序
        // "Duplicate -1 覆盖 [0,W) 再 MTE2 装载 [0,c)" 两管道对 UB 同址写,
        // 事件对在末核三行长队列(延伸分支:行5+零行6,7+cleanup 同 outQueue_)
        // 下偶发失配。改为区域划分:[0,c) MTE2 独占装载,[c,W) V 独占 -1 填充
        // —— 消除同址写,语义不变(ws 行 [validC,c) 本就为 -1 尾)。
        if (validC > 0) {
            // P1 双读合并:[0,c) 已在 dedup 阶段装载并消费,此处只补 [c,W) -1 填充
            PipeBarrier<PIPE_V>();
            Duplicate(outI32[c], constInfo_.INVALID_IDX, W - c);
            PipeBarrier<PIPE_V>();
        } else {
            // validC==0:无 ws 装载(全零批 ws 为 0 尺寸),整行 -1 + 身份 patch
            Duplicate(outI32, constInfo_.INVALID_IDX, W);
            PipeBarrier<PIPE_V>();
        }
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        if (nNew > 0) {
            for (uint32_t k = 0; k < nNew; k++) {
                outI32.SetValue(validC + k, auxI32.GetValue(k));
            }
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopyPad(candidatesOutGm_[r * W], outI32, {1, static_cast<uint16_t>(W * sizeof(int32_t)), 0, 0});
        if (useMask) {
            auxI32.SetValue(64, static_cast<int32_t>(validC + nNew));
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(aslkOutGm_[r], auxI32[64], {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        } else {
            auxI32.SetValue(64 + (r - rBegin), static_cast<int32_t>(validC + nNew));
        }
        // 2026-09-24 竞态第六例根修(R≥25 多行核 NONDET 全量判决 + tail 延伸核行5
        // 污染统一解释):行间复用 outI32 缺 MTE3 排空 —— 本行 DataCopyPad(MTE3
        // 读 outI32)未落地时,下一行 validC>0 走 MTE2 装载覆写 [0,c)、validC==0
        // 走 V Duplicate 覆写全行。覆写内容不同 ⇒ 三种签名:混入邻行内容(aime
        // -1计数 0→0)/整行错位(svc 差4096)/-1 洪泛 PARTIAL(tail600 计数漂移)。
        // 修复 = 生产 A3 同族先例(lightning_indexer_service_vector:687 循环内
        // DataCopyPad 后 SetWaitFlag<MTE3_V> 保下一轮 UB 复用;本文件入口三件套
        // 同款事件对)。两条覆写路径必须两个事件都加:只加 MTE3_MTE2 不序 V 写
        // (EmitIdentityRows 坑 #45"事件类不可靠"疑为只试单事件对的误诊——该处
        // 唯一覆写路径是 V,单加 MTE3_V 即应有效;PIPE_ALL 有效只因其为全超集)。
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    if (!useMask && rEnd > rBegin) {
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(aslkOutGm_[rBegin], auxI32[64],
                    {1, static_cast<uint16_t>((rEnd - rBegin) * sizeof(int32_t)), 0, 0});
    }
}
} // namespace LIKernel
#endif
