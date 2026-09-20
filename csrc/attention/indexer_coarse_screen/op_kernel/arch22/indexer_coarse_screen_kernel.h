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
 * \file indexer_coarse_screen_kernel.h
 * \brief PIVOT 粗筛内核(生产 lightning_indexer arch22 MIX 克隆 + 3-delta):
 *   Delta-A M1 组均值代理(AIV 全局预阶段,service_vector::ProcessGroupMean):
 *     q_bar/w_bar = 组内行按 row_weights 加权均值(fp32 累加,一次舍入),连同
 *     proxyCum=[1..R] 写 workspace;收尾 PipeBarrier<MTE3> + SyncAll + 预置
 *     syncV1C1×2(§4.11 同步模式),AIC 首个 matmul 的 WaitFlag 由此放行。
 *   主 pass = 生产原生 TND + PA_BSND 路径:query=q_bar(每请求 1 行 proxy)、
 *     weights=w_bar、key 经块表直读连续逻辑位置(KeyNd2NzForPA 原生语义)、
 *     aslk=粗筛域上界(绝对值)、sparseCount=coarseCount=4096 → over-2K 骨架
 *     (v11 双累积 2-list 归并,NPU 17/17 终验同款);输出候选 = 0-based 逻辑
 *     位置(分数降序,-1 终止),无候选列号间接层。
 *   Delta-B M2.5 窗口注入(AIV 全局后阶段,service_vector::ProcessWindow,hasWindow
 *     门控):win = [aslk-(g-1), aslk+aslq差分) 去重并入候选行(有效前缀 compact),
 *     尾部 -1 补齐到 outW;第二输出 aslk' = 每行有效候选数。
 *   LD 跨核归并删除:over-2K 下 SplitCore 恒整请求单核(s2End=s2BaseNum-1),
 *     isLD 恒 false,ProcessDecode 永不触发(原样保留属死代码,故裁剪)。
 */

#ifndef INDEXER_COARSE_SCREEN_KERNEL_H
#define INDEXER_COARSE_SCREEN_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../indexer_coarse_screen_common.h"
#include "indexer_coarse_screen_service_vector.h"
#include "indexer_coarse_screen_service_cube.h"

namespace LIKernel {
using namespace IndexerCoarseScreenCommon;
using namespace IndexerCoarseScreenServiceVec;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;      // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;       // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL;     // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;    // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;  // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
};

template <typename LIT>
class IndexerCoarseScreenKernel {
public:
    __aicore__ inline IndexerCoarseScreenKernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *qBar, __gm__ uint8_t *wBar, __gm__ uint8_t *key,
                                __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *candidatesOut, __gm__ uint8_t *aslkOut,
                                __gm__ uint8_t *workspace,
                                const IndexerCoarseScreenTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    // 编译期条件选择模板第二个参数的类型，直接声明W_T
    using W_T = typename IndexerCoarseScreenTypeTraits<Q_T,
                                                typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    using MM1_OUT_T = float;

    IndexerCoarseScreenServiceCube<LIT> matmulService;
    IndexerCoarseScreenServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t M_BASE_SIZE = 512;
    static constexpr uint32_t S2_BASE_SIZE = 512;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SPARSE_COUNT_8K = 8192;
    static constexpr uint32_t BLOCK_CUBE_SIZE = 16;

    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;

    // ================================Global Buffer区=================================
    // caller 输入(M1 / 窗口阶段消费,仅 AIV)
    GlobalTensor<uint32_t> callerSeqLenGmQ; // caller aslq [R] 累计(差分 own_tokens)
    GlobalTensor<uint32_t> actualSeqLengthsGm; // aslk [R] 粗筛域上界(绝对值)
    // M1 输出 / 主 pass 输入(workspace)
    GlobalTensor<Q_T> qBarGm;           // q_bar [R,H,Dh](caller 输入,主 pass 的 query)
    GlobalTensor<Q_T> wBarGm;           // w_bar [R,H](caller 输入,主 pass 的 weights)
    GlobalTensor<int32_t> candidatesWsGm; // 窗口模式主 pass 候选中转 [R,sparseCount]
    // 主 pass 消费(克隆结构保留)
    GlobalTensor<int32_t> blockTableGm;

    // 输出
    GlobalTensor<int32_t> candidatesOutGm; // 输出 candidates [R,outW]
    GlobalTensor<int32_t> aslkOutGm;    // 输出 aslk' [R]

    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    IndexerCoarseScreenCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    IndexerCoarseScreenCommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const IndexerCoarseScreenTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, IndexerCoarseScreenCommon::SplitCoreInfo &info);
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo);
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    template <typename GMT>
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GMT &actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::InitTilingData(const IndexerCoarseScreenTilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    // 每请求 1 行 proxy(q_bar),主 pass 的 s1 = 1
    constInfo.qSeqSize = 1ULL;
    // 无窗口掩码、无 values 输出 → attenMaskFlag/returnValue 恒 false
    constInfo.attenMaskFlag = false;
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount; // = coarseCount(输出候选宽,4096)
    constInfo.preTokens = INT64_MAX;
    constInfo.nextTokens = INT64_MAX;
    constInfo.returnValue = false;

    // coarse_screen 专属(M1 组均值 / M2.5 窗口)
    constInfo.groupSize = tilingData->groupSize;
    constInfo.windowG = tilingData->windowG;
    constInfo.outW = tilingData->outW;
    constInfo.hasWindow = tilingData->hasWindow; // 原值(0/1/2)直传,bool 折叠会废掉 debug dump

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
    if (LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true; // proxyCum 累计
    }
    if (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true; // PA_BSND → false,aslk 绝对值
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.isSparseCountOver2K = (constInfo.sparseCount <= BASE_TOPK) ? false : true;

    constInfo.s1BaseSize = constInfo.isSparseCountOver2K ? SPARSE_COUNT_8K / constInfo.sparseCount * 2 : 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;
    constInfo.mBaseSizeAlign = IndexerCoarseScreenCommon::Align(constInfo.mBaseSize, BLOCK_CUBE_SIZE);
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                        __gm__ uint8_t *actualSeqLengths)
{
    // caller aslq:仅 M1/窗口阶段消费(差分 own_tokens);主 pass 用 proxyCum
    if (actualSeqLengthsQ == nullptr) {
        constInfo.actualLenQDims = 0;
    } else {
        constInfo.actualLenQDims = constInfo.batchSize;
        callerSeqLenGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }
    // aslk:主 pass S2 域上界(PA 布局 → 绝对值语义)
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename LIT>
template <typename GMT>
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetActualSeqLen(uint32_t bIdx,
                                                           uint32_t actualLenDims, bool isAccumSeq,
                                                           GMT &actualSeqLengthsGm,
                                                           uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx,
                                                         uint32_t &actS1Size, uint32_t &actS2Size)
{
    // 每请求恒 1 行 proxy(q_bar)→ actS1Size 用常量 1,不读 proxyCum:
    // SplitCore 在 Init 期运行,彼时 proxyCum(M1 产物)尚未写,读 workspace 是未初始化值。
    // proxyCum 只在 ProcessMain 之后消费(CalcRunInfo 输出前缀和 / DealActSeqLenIsZero)。
    actS1Size = 1U;
    actS2Size =
        GetActualSeqLen(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2, actualSeqLengthsGm, constInfo.kSeqSize);
}

template <typename LIT>
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                   uint32_t actS2Size)
{
    if (actS2Size == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = s1Offset + validS2LenBase + constInfo.s1BaseSize;
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2Size));
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = constInfo.isSparseCountOver2K
                      ? (actS2Size > 0 ? 1 : 0)
                      : CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isSparseCountOver2K
                      ? (actS2Size > 0 ? 1 : 0)
                      : GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2Size);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline IndexerCoarseScreenKernel<LIT>::SplitCore(uint32_t curCoreIdx,
                                                         uint32_t &coreNum, IndexerCoarseScreenCommon::SplitCoreInfo &info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            s2Loop = constInfo.isSparseCountOver2K ? (actS2Size > 0 ? 1 : 0) : s2BaseNum;
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = constInfo.isSparseCountOver2K
                               ? s2BaseNum - 1
                               : s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

                    if (coreIdx == curCoreIdx) {
                        info.isEmptyRange = false; // 拥有任务段的核
                        // LD 跨核归并已裁剪:coarse_count 强制 >2048(over-2K),
                        // over-2K 恒整请求单核(s2End = s2BaseNum-1),无跨核续核需要归并
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize -1) {
                            info.bN2End = constInfo.batchSize -1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = constInfo.batchSize;                 // 每请求 1 行 proxy:总行数 == R
            uint32_t tBase = bIdx;                                // prefix 硬编码(同 CalcRunInfo,零 GM 读)
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = bIdx * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                                           s1Idx * constInfo.kHeadNum * constInfo.sparseCount + // B轴、S1轴偏移
                                           n2Idx * constInfo.sparseCount;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::Init(__gm__ uint8_t *qBar,
                                            __gm__ uint8_t *wBar, __gm__ uint8_t *key,
                                            __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *candidatesOut,
                                            __gm__ uint8_t *aslkOut,
                                            __gm__ uint8_t *workspace, const IndexerCoarseScreenTilingData *__restrict tiling,
                                            TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 计算分核
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    pipe = tPipe;
    // workspace 内存排布(M1 出核):|mm1ResGm(存S,DB/核)|candidatesWsGm [R,sparseCount](窗口模式)|
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.mBaseSizeAlign * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    auto alignGm = [](uint64_t bytes) -> uint64_t {
        return (bytes + GM_ALIGN_BYTES - 1) / GM_ALIGN_BYTES * GM_ALIGN_BYTES;
    };
    // M1 出核(2026-09-17):q_bar/w_bar 为 caller 输入,直绑(原生 AIC 读 caller
    // 输入模式);workspace 只剩 mm1 + candidatesWs(窗口中转)。proxyCum 删除
    // (prefix 已硬编码 bIdx),qBar/wBar 相关 M1 位视图随 M1 一并移除。
    qBarGm.SetGlobalBuffer((__gm__ Q_T *)qBar);
    wBarGm.SetGlobalBuffer((__gm__ Q_T *)wBar);
    __gm__ uint8_t *wBarPtr = wBar;
    if (constInfo.hasWindow) {
        candidatesWsGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
        offset += alignGm(constInfo.batchSize * constInfo.sparseCount * sizeof(int32_t));
    }

    candidatesOutGm.SetGlobalBuffer((__gm__ int32_t *)candidatesOut);
    aslkOutGm.SetGlobalBuffer((__gm__ int32_t *)aslkOut);
    blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        // M1 出核:窗口阶段仅需 caller aslq/aslk/候选中转/输出(组均值由 caller 传)
        GlobalTensor<int32_t> dbgMm1;
        dbgMm1.SetGlobalBuffer((__gm__ int32_t *)workspace);
        vectorService.InitCoarseGlobalTensor(callerSeqLenGmQ, actualSeqLengthsGm, candidatesWsGm,
                                             candidatesOutGm, aslkOutGm, dbgMm1);
        // 主 pass 直写目标:窗口模式先落 workspace(行距 sparseCount),否则直写输出(行距 outW=coarseCount)
        GlobalTensor<int32_t> mainPassOut = constInfo.hasWindow ? candidatesWsGm : candidatesOutGm;
        // 主 pass weights = w_bar(caller 输入);W_T 视图(DT_W_FLAG=true 时为 float)
        GlobalTensor<W_T> wBarW;
        wBarW.SetGlobalBuffer((__gm__ W_T *)wBarPtr);
        vectorService.InitVec1GlobalTensor(mm1ResGm, wBarW, mainPassOut);
    } else {
        matmulService.InitParams(constInfo);
        // 主 pass query = q_bar(M1 输出);key = 原始 PA cache(块表直读)
        GlobalTensor<Q_T> cubeQuery = qBarGm;
        GlobalTensor<K_T> paKeyGm;
        paKeyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, paKeyGm, cubeQuery, mm1ResGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::CalcRunInfo(uint32_t loop,
                                                         uint32_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        IndexerCoarseScreenCommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, IndexerCoarseScreenCommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        uint64_t actualSeqKPrefixSum;
        if constexpr (LAYOUT_T == LI_LAYOUT::TND) {
            // 2026-09-17 prefix 硬编码:每请求恒 1 行 proxy ⟹ prefix(bIdx) == bIdx。
            // 旧版读 proxyCum[bIdx-1](M1 写于 pair bIdx-1,读于 pair bIdx —— 跨对 GM 读,
            // NPU mm1 判决实证 AIC3 读到 prefix=4 即 row4 的分):种子只闭环本对依赖,
            // SyncAll 不足以保证跨对可见时序。硬编码后计算路径零 proxyCum 读,跨对依赖清零。
            actualSeqQPrefixSum = runInfo.bIdx;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
            actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        uint64_t tndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D(PA 模式 key 走块表直读,tensorKeyOffset 仅非 PA 克隆分支消费)
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        keyCoreOffset = tndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
                              runInfo.n2Idx * constInfo.sparseCount;
    }
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum
    * constInfo.headDim;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::Process()
{
    if (usedCoreNum == 0) {
        // 全零批(所有行粗筛域空,如 prefill-PIVOT 的请求首组单独成批):窗口模式
        // 仍须发恒等语义行(池空,own tokens [0, upper+own) 全进精筛域)—— 主 pass
        // 未运行,ws 无 DealActSeqLenIsZero 的 -1 预填,各 AIV 协作补填后由 0 号核
        // 走已验证的 ProcessWindow(validC==0 分支:免去重,own 全追加)。
        // 纯粗筛/dump 模式保持原清理路径。
        if (constInfo.hasWindow == 1U && constInfo.batchSize > 0) {
            // 0 号 AIV 走已验证窗口路径;validC==0 行现全程零 ws 访问
            // (2026-09-21:此前 InitGlobalMemory 填 ws 方案挂起 —— 零批 tiling
            //  的 workspace 为 0 尺寸,任何 ws 触碰都不可行)。
            if ASCEND_IS_AIV {
                if (tmpBlockIdx == 0) {
                    vectorService.ProcessWindow(pipe, 0, constInfo.batchSize);
                }
            }
            return;
        }
        ProcessInvalid();
        return;
    }
    // 阶段门控调试:hasWindow==3 只跑 M1+dump(主 pass/窗口全跳,须双核同步跳过防
    // 种子 flag 悬空);==4 跑 M1+主 pass+dump(跳窗口)。用于多 chunk fault 的阶段二分。
    if (constInfo.hasWindow != 3U) {
        ProcessMain();
    } else {
        if ASCEND_IS_AIC {
            return; // 无主 pass 时 AIC 无事可做,直接退出(不参与任何同步)
        }
    }
    if ASCEND_IS_AIV {
        // M2.5 窗口注入(全局后阶段):行范围 = 本对 SplitCore 请求范围(偶 AIV 承担,
        // 与 M1 同款配对 —— 行 r 的窗口读者 = 主 pass CopyOut 写者 AIV 2r,同核有序)
        uint32_t wBegin = splitCoreInfo.isEmptyRange ? 0U : splitCoreInfo.bN2Start;
        uint32_t wEnd = splitCoreInfo.isEmptyRange ? 0U : splitCoreInfo.bN2End + 1U;
        if (tmpBlockIdx % 2 == 1) {
            wBegin = wEnd; // 奇 AIV(对的第 2 个)不承担窗口读写
        }
        vectorService.ProcessWindow(pipe, wBegin, wEnd);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.outW * constInfo.kHeadNum;
        uint64_t singleCoreSize =
            IndexerCoarseScreenCommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = candidatesOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
        uint64_t totalAslkSize = constInfo.batchSize;
        uint64_t aslkSingleCoreSize =
            IndexerCoarseScreenCommon::Align((totalAslkSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(int32_t));
        uint64_t aslkBase = tmpBlockIdx * aslkSingleCoreSize;
        if (aslkBase < totalAslkSize) {
            uint64_t aslkDeal =
                (aslkBase + aslkSingleCoreSize <= totalAslkSize) ? aslkSingleCoreSize : totalAslkSize - aslkBase;
            GlobalTensor<int32_t> aslkTarget = aslkOutGm[aslkBase]; // 具名左值(InitGlobalMemory 形参为非常量引用)
            AscendC::InitGlobalMemory(aslkTarget, aslkDeal, 0);
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回(同 lightning_indexer:任务检查最先,无任务核不参与任何同步)
        return;
    }

    if ASCEND_IS_AIV {
        // M1 已出核(q_bar/w_bar 为 caller 输入);此处预置 syncV1C1×2 种子 flag。
        // 2026-09-16:种子管道由 MTE2 改绑 MTE3 —— qBar/wBar 落地走 MTE3 管道,
        // MTE2 语义种子(原生无 M1 的写法)不等 MTE3 队列,rows≥1 的 M1 写出
        // 相对 AIC 首轮读取存在窗口(NPU 实测 rows≥1 间歇坏、row 0 恒好);
        // MTE3 语义种子保证 flag 置位时 qBar 已在 GM。
        vectorService.AllocEventID();
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    IndexerCoarseScreenCommon::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.isEmptyRange ? 1U : splitCoreInfo.bN2Start;
         !splitCoreInfo.isEmptyRange && bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessBaseBlock(uint32_t loop,
                                                         uint64_t s2LoopIdx, IndexerCoarseScreenCommon::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<IndexerCoarseScreenCommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}
} // namespace LIKernel
#endif // INDEXER_COARSE_SCREEN_KERNEL_H
