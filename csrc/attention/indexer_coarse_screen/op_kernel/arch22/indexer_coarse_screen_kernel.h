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
 * \brief PIVOT 粗筛内核(生产 lightning_indexer arch22 MIX 克隆裁剪版):
 *   主 pass = 原生 TND + PA_BSND 路径:query=q_bar(caller 输入,每请求 1 行
 *     proxy)、weights=w_bar、key 经块表直读连续逻辑位置、aslk=粗筛域上界
 *     (绝对值)、coarseCount=4096 → over-2K 单次归并骨架(SortAll+MergeSort);
 *     输出候选 = 0-based 逻辑位置(分数降序,-1 终止)。q_bar/w_bar 为 caller
 *     输入直绑(workspace 仅 mm1 + candidatesWs)。
 *   窗口注入(M2.5,AIV 全局后阶段,hasWindow 门控):win =
 *     [aslk-(g-1), aslk+aslq差分) 去重并入候选行(有效前缀 compact),尾部 -1
 *     补齐到 outW;第二输出 aslk' = 每行有效候选数。
 *   LD 跨核归并已裁剪:over-2K 下 SplitCore 恒整请求单核,isLD 恒 false。
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
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> partialTopkGm; // M3:跨核 fragment partial(经 service 使用)  // 存放S

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;
    // 整批总块数(SplitCore 每核独立遍历全部行计算,各核天然同值)。
    // 0 = 整批无块(所有行粗筛域空):host 无法预知(upper 是 device 张量,禁 D2H
    // 同步),tiling 的 usedCoreNum 恒 ≥1 → 全零批判据只能在 kernel 侧算。
    // 2026-09-24 修复:整批无块时原版输出整批无人写(回收垃圾,refine 连锁吃垃圾)。
    uint32_t totalBlockNum = 0U;
    // M3:全局块流切分参数(SplitCore 填,CalcRunInfo 下发)
    uint32_t splitCap_ = 0U;
    uint32_t splitDeal1More_ = 0U;

    IndexerCoarseScreenCommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    IndexerCoarseScreenCommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const IndexerCoarseScreenTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, IndexerCoarseScreenCommon::SplitCoreInfo &info);
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
    // 每请求 1 行 proxy(q_bar),主 pass 的 s1 = 1(qSeqSize 字段已随 BSND 死支删除)
    // 无窗口掩码 → attenMaskFlag 恒 false(相关掩码分支已删)
    constInfo.attenMaskFlag = false;
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount; // = coarseCount(输出候选宽,4096)

    // coarse_screen 专属(M2.5 窗口)
    constInfo.groupSize = tilingData->groupSize;
    constInfo.windowG = tilingData->windowG;
    constInfo.outW = tilingData->outW;
    constInfo.hasWindow = tilingData->hasWindow; // 原值(0/1/2)直传,bool 折叠会废掉 debug dump
    constInfo.proxyMode = tilingData->proxyMode;

    constInfo.outputLayout = LAYOUT_T; // 输出和输入形状一致
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
    // 注:唯一调用点传 isAccumSeqS2(PA_BSND → false),累减分支为克隆保留形态
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
__aicore__ inline uint32_t IndexerCoarseScreenKernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        // M3U:总块数与执行粒度一致 —— 旧 over-2K 记账每行 1 块(s2Loop≡1 时代),
        // fine-split 下执行侧 s2Loop=s2BaseNum,记账不改则 totalBlockNum 少算
        // s2BaseNum 倍:minBlockPerCore=0 → usedCoreNum 被改写为行数,每核 1 块
        // 却经非末 tile 的 s2LoopEnd=s2BlockNum-1 展开成整行(末核再叠尾零调整
        // 跨行全扫)→ 3x 冗余 507014 / L2K 档 4x 慢。
        s2BaseNum = (actS2Size > 0) ? CeilDiv(actS2Size, constInfo.s2BaseSize) : 0;
        totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename LIT>
__aicore__ void inline IndexerCoarseScreenKernel<LIT>::SplitCore(uint32_t curCoreIdx,
                                                         uint32_t &coreNum, IndexerCoarseScreenCommon::SplitCoreInfo &info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    // (写入成员:全零批判据供 Process 使用,各核独立计算结果一致)
    totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;
    splitCap_ = minBlockPerCore;
    splitDeal1More_ = deal1MoreBlockCoreNum;

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
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            // M3:over-2K 亦按 s2 子块入全局流(旧=每行 1 单元 → 行内串行台阶)
            s2Loop = s2BaseNum;
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (coreDealBlockCnt == 0U && lastGS1RemainBlockCnt == 0U) {
                    // M3R-A:零预算核(blocks < 核数,粒度化后 minBlockPerCore=0)
                    // 干净置空 —— 旧逻辑误走赋值分支,s2End = s2Idx-1 uint 下溢,
                    // curCoreIdx 命中时 s2LoopEnd=0xFFFFFFFF 野读 507015。
                    if (coreIdx == curCoreIdx) {
                        info.isEmptyRange = true;
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    coreDealBlockCnt =
                        coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                    continue;
                }
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;

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
                    // M3R-B:预算未满,逐 unit 记账(s2Loop≡1 时代余量恒 1,
                    // += s2Remain 与 += 1 等价;粒度化后必须按单位累计,否则
                    // lastGS1 平方超累计 → s2End 负下溢)
                    lastGS1RemainBlockCnt += 1U;
                    s2Idx++;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        // TND:每请求 1 行 proxy,总行数 == R,prefix 硬编码(同 CalcRunInfo,零 GM 读)
        uint32_t tBase = bIdx;
        uint32_t s1Count = tempLoopInfo.actS1Size;

        for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
            uint64_t indiceOutOffset =
                (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // T轴、s1轴偏移
                n2Idx * constInfo.sparseCount;                                 // N2轴偏移
            vectorService.CleanInvalidOutput(indiceOutOffset);
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
    // 注:MODE2 flag id 空间极小(per-pair 分 id 实测无效),维持全 pair 共 id

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
    // 2026-09-17 M1 出核:q_bar/w_bar caller 直绑。2026-10-08 M1a:proxy_mode=1
    // 时 query/weights 入参为 raw [N,H,Dh]/[N,H],AIV 预阶段组均值写 workspace
    // qBarWs/wBarWs(与 host tiling 同序:追加在 candidatesWs 之后),主 pass 的
    // 消费点(qBarGm/wBarGm 及其下游 cube queryGm_/vector weightsGm)整链改绑。
    if (constInfo.hasWindow) {
        candidatesWsGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
        offset += alignGm(constInfo.batchSize * constInfo.sparseCount * sizeof(int32_t));
    }
    __gm__ uint8_t *wBarEff = wBar; // DoScale 的 W_T 视图基址(mode1 = wBarWs 段)
    if (constInfo.proxyMode) {
        qBarGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + offset));
        offset += alignGm(constInfo.batchSize * constInfo.qHeadNum * constInfo.headDim * sizeof(Q_T));
        wBarGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + offset));
        wBarEff = workspace + offset;
        offset += alignGm(constInfo.batchSize * constInfo.qHeadNum * sizeof(Q_T));
    } else {
        // mode 0(兼容/probe A/B):caller 预算 q̄/w̄ 直绑,原路径逐位保持
        qBarGm.SetGlobalBuffer((__gm__ Q_T *)qBar);
        wBarGm.SetGlobalBuffer((__gm__ Q_T *)wBar);
    }

    candidatesOutGm.SetGlobalBuffer((__gm__ int32_t *)candidatesOut);
    aslkOutGm.SetGlobalBuffer((__gm__ int32_t *)aslkOut);
    blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        // M1 出核:窗口阶段仅需 caller aslq/aslk/候选中转/输出(组均值由 caller 传)
        GlobalTensor<int32_t> dbgMm1;
        // M3:partial topk 槽(48 = 24 pair × 2 seq;槽 = 8 innerS1 × 4096×2 fp32)
        partialTopkGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset += alignGm(48ULL * (constInfo.batchSize * constInfo.kHeadNum * 4096U * 2U + 16U) * sizeof(float));
        dbgMm1.SetGlobalBuffer((__gm__ int32_t *)workspace);
        vectorService.InitCoarseGlobalTensor(callerSeqLenGmQ, actualSeqLengthsGm, candidatesWsGm,
                                             candidatesOutGm, aslkOutGm, dbgMm1);
        // M1a 预阶段视图:raw = caller 输入原样(mode1 才消费);q̄/w̄ 写出目标 =
        // 上面的 mode 选择结果(mode1 = workspace 段,与主 pass 读完全一致)
        vectorService.InitProxyGlobalTensor(qBar, wBar, qBarGm, wBarGm);
        // 主 pass 直写目标:窗口模式先落 workspace(行距 sparseCount),否则直写输出(行距 outW=coarseCount)
        GlobalTensor<int32_t> mainPassOut = constInfo.hasWindow ? candidatesWsGm : candidatesOutGm;
        // 主 pass weights = w̄(必须跟随 wBarGm 的 mode 绑定:mode1 = workspace
        // 段,mode0 = caller 输入)。曾误绑 raw wBar —— mode1 下 DoScale 读到
        // raw 行而非 w̄,分数全错(2026-10-08 probe_m1a 判决)。
        GlobalTensor<W_T> wBarW;
        wBarW.SetGlobalBuffer((__gm__ W_T *)wBarEff);
        vectorService.InitVec1GlobalTensor(mm1ResGm, wBarW, mainPassOut, partialTopkGm);
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
    uint32_t s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
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

    runInfo.unitStartSelf = static_cast<uint32_t>(aiCoreIdx) * splitCap_ +
        (static_cast<uint32_t>(aiCoreIdx) < splitDeal1More_ ? static_cast<uint32_t>(aiCoreIdx) : splitDeal1More_);
    runInfo.unitCap = splitCap_;
    runInfo.deal1More = splitDeal1More_;
    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);

    if (runInfo.isFirstS2InnerLoop) {
        // 2026-09-17 prefix 硬编码:每请求恒 1 行 proxy ⟹ prefix(bIdx) == bIdx。
        // 旧版读 proxyCum[bIdx-1](M1 写于 pair bIdx-1,读于 pair bIdx —— 跨对 GM 读,
        // NPU mm1 判决实证 AIC3 读到 prefix=4 即 row4 的分):种子只闭环本对依赖,
        // SyncAll 不足以保证跨对可见时序。硬编码后计算路径零 proxyCum 读,跨对依赖清零。
        uint64_t actualSeqQPrefixSum = runInfo.bIdx;
        uint64_t actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx - 1);
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
    if ASCEND_IS_AIV {
        // M1a 组均值预阶段(proxyMode 门控,mode0 首行 return):行按 pair 分区,
        // 行由 pair 偶 AIV 独占计算(单写者,防同址并发写互踩),写毕置 MTE3
        // 管道种子 flag 背书 —— 不用 SyncAll:SyncAll 不保证 AIC 侧 GM 可见性
        // (def.cpp 2026-09-17 出核注记,NPU 实测竞态复证)。
        // bN2End 为闭区间,转半开 [bN2Start, bN2End+1) —— 曾按半开直传导致
        // 单行 pair(最常见)bN2Start==bN2End,预阶段整体死代码(零分总根源)
        vectorService.ProcessGroupMean(splitCoreInfo.bN2Start, splitCoreInfo.bN2End + 1U,
                                       splitCoreInfo.isEmptyRange);
    }
    if (usedCoreNum == 0 || totalBlockNum == 0) {
        // 全零批(所有行粗筛域空,如 prefill-PIVOT 的请求首组单独成批):窗口模式
        // 2026-09-24 起真实可达:判据 = totalBlockNum==0(SplitCore 每核独立算,
        // 同值同步分支,全 AIV 仍同时到屏障,无死锁);usedCoreNum==0 为 host 侧
        // 防御(host tiling 恒置 ≥1,保留)。修复前此场景输出整批无人写(回收垃圾)。
        // 仍须发恒等语义行(池空,own tokens [0, upper+own) 全进精筛域)—— 主 pass
        // 未运行,ws 无 DealActSeqLenIsZero 的 -1 预填,各 AIV 协作补填后由 0 号核
        // 走已验证的 ProcessWindow(validC==0 分支:免去重,own 全追加)。
        // 纯粗筛/dump 模式保持原清理路径。
        if (constInfo.hasWindow == 1U && constInfo.batchSize > 0) {
            // 全零批走已验证窗口路径(validC==0 行全程零 ws 访问)。
            // 关键(2026-09-21 挂起根因):ProcessWindow 内部首段有 SyncAll() 跨核
            // 屏障 —— 必须与正常路径同形态:全部 AIV 进入(空范围也到屏障),
            // 行按偶 AIV 均分、奇 AIV 空范围;单核进入 = 独自等全员屏障 = 死锁。
            if ASCEND_IS_AIV {
                uint32_t aivCoreNum = GetBlockNum() * 2;
                uint32_t per = (constInfo.batchSize + aivCoreNum / 2 - 1) / (aivCoreNum / 2);
                uint32_t idx = tmpBlockIdx / 2;
                uint32_t wB = 0U;
                uint32_t wE = 0U;
                if (tmpBlockIdx % 2 == 0) {
                    wB = idx * per;
                    wE = wB + per > constInfo.batchSize ? constInfo.batchSize : wB + per;
                }
                vectorService.ProcessWindow(pipe, wB, wE);
            }
            return;
        }
        ProcessInvalid();
        return;
    }
    ProcessMain();
    if ASCEND_IS_AIV {
        // M2.5 窗口注入。2026-09-21 终版:恢复 pd 验证的 splitCoreInfo 配对分区
        // (行 r 的窗口读者 = 主 pass CopyOut 写者同核,同核 MTE3→MTE2 有序 ——
        // 这是消除跨对 GM 可见性时序的原始设计,aivPairs 均分切片方案在 16 卡
        // 生产冷启动首次调用时间歇输出垃圾 = 打破配对的竞态签名)。
        // 空请求行(upper=0 ⇒ 块数 0,被 bN2Start 跳过、原版无人写出)由 pair 0
        // 在进 ProcessWindow 前直写恒等行 [0, upper+own)(ws-free,零同步依赖,
        // 主 pass 从未写这些行的 ws,无可见性问题)。
        uint32_t wBegin = splitCoreInfo.isEmptyRange ? 0U : splitCoreInfo.bN2Start;
        uint32_t wEnd = splitCoreInfo.isEmptyRange ? 0U : splitCoreInfo.bN2End + 1U;
        if (tmpBlockIdx % 2 == 1) {
            wBegin = wEnd; // 奇 AIV(对的第 2 个)不承担窗口读写
        }
        // M3I:Flush 移到窗口前 —— marker-poll 自带跨核可见性,不依赖窗口内
        // SyncAll;且窗口内部 pipe->Reset() 会作废全部 UB 句柄,Flush 的
        // outQueue_/tmpUb_ 在其后分配即踩非法地址(MTE invalid address 实证)。
        // M3F2:fold 重启用 —— 炸机根因 = MrgSort elementLengths 段长越界
        // (官方约束 [0,4095],8B 单位;旧体传 8192 对单段),非时序脆弱。
        // FoldMerge4(4×2048)修复后重启用跨核折叠。
        vectorService.FlushDeferredMerges();
        // M3F3:窗口行集 = 行位图(读者=写者同核);[0,batchSize) 全域扫描 + 位图过滤
        vectorService.ProcessWindow(pipe, 0U, constInfo.batchSize, true);
        // 空请求行补写放 ProcessWindow 之后:其内部 pipe->Reset()+InitBuffer
        // 已初始化 winOut/winAux,EmitIdentityRows 直接复用(零 ws 依赖)。
        // M3F9:空域行(upper==0)已由 dedup 的 validC==0 恒等分支冗余覆盖,
        // EmitIdentityRows 整条路径删除(M3F8 位图私有性缺陷:他核行不可见)
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
        // 2026-09-24 修复(全零批 gate 12 项 FAIL 根因):InitGlobalMemory 实测
        // 每 core 多发 init 只有部分落地(指纹填充:同核 [C 区间0, aslk] 两发,
        // 区间0 必丢;"首失/末胜"两种模型 NPU 不可区分,错位换核复测证明失效
        // 跟地址无关)。防护 = 每核先向 workspace(mm1 段,全零路径不使用)打
        // 一发 dummy init 吸收失效,再写本核真正的 C 区间 —— 首失模型下 dummy
        // 顶枪、末胜模型下 dummy 被丢弃,两种模型下 C 必落地;48 核原始映射
        // i→区间 i,覆盖无洞。
        // aslk 清理(2026-09-28 改标量 SetValue):host at::zeros 在生产 GLOBAL
        // 捕获态触发 EE1016 同步拷贝(整网 AIME 崩溃主嫌)已回退;标量写不走
        // InitGlobalMemory 槽位,无同核多发丢失问题。死路径,R ≤ 数千,性能无谓。
        GlobalTensor<MM1_OUT_T> dummyTarget = mm1ResGm[0]; // 具名左值(InitGlobalMemory 形参为非常量引用)
        AscendC::InitGlobalMemory(dummyTarget, 1U, static_cast<MM1_OUT_T>(0)); // dummy(吸收首失)
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = candidatesOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
        if (tmpBlockIdx == 0) {
            for (uint32_t i = 0; i < constInfo.batchSize; i++) {
                aslkOutGm.SetValue(i, 0);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void IndexerCoarseScreenKernel<LIT>::ProcessMain()
{
    // M3:usedCoreNum 是行基核数,R 小于 pair 数时早退会杀死已分到子块的 pair 的
    // AIC(其 AIV 将永等 C1V1 信用 → 挂死)。空范围由 isEmptyRange 自然跳过;
    // 空 pair AIC 的末尾 drain 有 4 颗种子信用可消费,不挂。

    if ASCEND_IS_AIV {
        // M1 已出核(q_bar/w_bar 为 caller 输入);此处预置 syncV1C1×2 种子 flag。
        // 2026-09-16:种子管道由 MTE2 改绑 MTE3 —— qBar/wBar 落地走 MTE3 管道,
        // MTE2 语义种子(原生无 M1 的写法)不等 MTE3 队列,rows≥1 的 M1 写出
        // 相对 AIC 首轮读取存在窗口(NPU 实测 rows≥1 间歇坏、row 0 恒好);
        // MTE3 语义种子保证 flag 置位时 qBar 已在 GM。
        vectorService.AllocEventID();
        if ASCEND_IS_AIV {
            vectorService.ZeroMyMagics(); // M3B:清零本 AIV magic 块(防陈旧命中)
        }
        // M1a 预阶段全局屏障:各 AIV 写自有标记槽后轮询全员到齐,期间不放行
        // 种子置位。种子信用全局可互换(MODE2 flag id 空间极小,分 id 实测无效
        // 且 id=1 无序语义 —— AIC 侧时序快慢会改变结果 = 竞态铁证),唯有
        // "每个信用都全量背书"能封死 AIC 早读未落地 q̄/w̄(±v 零均值组首分数
        // 1.07e9 = workspace 残留的确定性签名,2026-10-09 NPU 实证)。
        if (constInfo.proxyMode) {
            // DIAG-B vectorService.PreStageBarrier(); // 仅 mode1:预阶段写才需全量背书
        }
        // 种子 flag 必须 pair 内两 AIV 各置一次(AIC wait 等两核到齐,缺一即挂死
        // —— 507014 实证)。屏障背书后,种子只承担块循环的流水配速。
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
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        // M3:fragment partial 全局可见屏障(空范围核亦到齐,同窗口路径纪律;
        // usedCoreNum 域内全体 AIV 一致进入,域外核不参与——两侧判据一致无死锁)
        // M3:延迟归并移至 ProcessWindow 的 SyncAll 之后(彼时乒乓已静止,
        // CCU 安全;主循环后直接 SyncAll 会撞乒乓尾巴 → 507015,实测三连)
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
