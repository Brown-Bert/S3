#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelLogSumExp {
public:
    __aicore__ inline KernelLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t bigCoreDataNum, uint32_t tileDataNum, uint32_t bigCoreNum, 
        uint32_t bigCoreCarryNum, uint32_t bigCoreFinallDealNum, uint32_t smallCoreDataNum, uint32_t smallCoreCarryNum, 
        uint32_t smallCoreFinallDealNum, uint32_t dataType, int64_t dim, bool keepDim, uint32_t blockSize, uint32_t ridOfNum, uint32_t dims, uint32_t dimSize,
        uint32_t dataSize, uint32_t index, uint32_t loop, int32_t j, int flag)
    {
        //考生补充初始化代码
        this->flag = flag;
        this->index_j = j;
        this->loop = loop;
        this->index = index;
        this->dataSize = dataSize;
        this->dimSize = dimSize;
        this->dims = dims;
        int32_t typeSize = 2;
        if (dataType == 0){
            typeSize = 4;
        }
        int32_t elementsPerBlock = 32 / typeSize;
        int32_t elementsPerRepeat = 256 / typeSize;
        int32_t firstMaxRepeat = dataSize / elementsPerRepeat;
        int32_t finalWorkLocalNeedSize =  (firstMaxRepeat + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        this->ridOfNum = ridOfNum;
        this->blockSize = blockSize;
        this->dataType = dataType;
        this->dim = dim;
        this->keepDim = keepDim;
        uint32_t aicoreIndex = GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * aicoreIndex;
        this->tileDataNum = tileDataNum;
        if (aicoreIndex < bigCoreNum){
            this->coreDataNum = bigCoreDataNum;
            this->coreCarryTimes = bigCoreCarryNum;
            this->coreFinallDataNum = bigCoreFinallDealNum;
        }else{
            this->coreDataNum = smallCoreDataNum;
            this->coreCarryTimes = smallCoreCarryNum;
            this->coreFinallDataNum = smallCoreFinallDealNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (aicoreIndex - bigCoreNum);
        }
        // xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex + index * dataSize, this->coreDataNum);
        if (this->dataType == 0 && this->dims == 3){
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex + index * dataSize, this->coreDataNum);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex, this->coreDataNum);
        }else if (this->dataType == 0 && this->dims == 4){
            if (j == -1){
                xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex + index * (dataSize), this->coreDataNum);
                yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex, this->coreDataNum);
            }else{
                xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex + index * (dataSize - ridOfNum) * loop, this->coreDataNum);
                yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex + index * loop, this->coreDataNum);
            }
        }else{
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + globalBufferIndex + index * dataSize, this->coreDataNum);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex + index, this->coreDataNum);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueMax, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));

        // pipe.InitBuffer(tempQueueHalf, BUFFER_NUM, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tempQueueFloat, BUFFER_NUM, this->tileDataNum * sizeof(float));
        // pipe.InitBuffer(dstQueueUint8, BUFFER_NUM, this->tileDataNum);

        pipe.InitBuffer(workQueue, BUFFER_NUM, finalWorkLocalNeedSize * sizeof(float));

    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        uint32_t loopCount = this->coreCarryTimes;
        this->processDataNum = this->tileDataNum;
        // AscendC::printf("loopCount = %d, tileDataNum = %d, coreFinallDataNum = %d, dataSize = %d\n", loop, this->tileDataNum, 
        //     this->coreFinallDataNum, this->dataSize);
        if (this->dataType == 0 && this->dims == 3){
            for (int32_t i = 0; i < loopCount; i++) {
                if ( i == this->coreCarryTimes - 1){
                    this->processDataNum = this->coreFinallDataNum;
                }
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }
        }else {
            if (this->dims < 4){
                for (int32_t i = 0; i < loopCount; i++) {
                    if ( i == this->coreCarryTimes - 1){
                        this->processDataNum = this->coreFinallDataNum;
                    }
                    CopyIn(i);
                    Compute(i);
                    CopyOut(i);
                }
                CopyIn();
                Compute();
                CopyOut();
            }else{
                if (this->index_j == -1){
                    for (int32_t i = 0; i < loopCount; i++) {
                        if ( i == this->coreCarryTimes - 1){
                            this->processDataNum = this->coreFinallDataNum;
                        }
                        CopyIn(i);
                        Compute(i);
                        CopyOut(i);
                    }
                }else{
                    for (int32_t i = 0; i < this->loop; ++i){
                        CopyIn(i);
                        Compute(i);
                        CopyOut(i);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        if (this->dataType == 0 && this->dims == 3){
            AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
            if (this->index != 0){
                AscendC::LocalTensor<DTYPE_X> xLocalTmp = inQueueY.AllocTensor<DTYPE_X>();
                AscendC::DataCopy(xLocalTmp, yGm[progress * this->tileDataNum], this->processDataNum);
                inQueueY.EnQue<DTYPE_X>(xLocalTmp);
            }
        }else if (this->dataType == 0 && this->dims == 4){
            if (this->index_j == -1){
                AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
                if (this->index != 0){
                    AscendC::LocalTensor<DTYPE_X> xLocalTmp = inQueueY.AllocTensor<DTYPE_X>();
                    AscendC::DataCopy(xLocalTmp, yGm[progress * this->tileDataNum], this->processDataNum);
                    inQueueY.EnQue<DTYPE_X>(xLocalTmp);
                }
            }else{
                AscendC::LocalTensor<DTYPE_X> xLocalTmp = inQueueY.AllocTensor<DTYPE_X>();
                for (int32_t i = 0; i < this->dataSize - this->ridOfNum; ++i){
                    AscendC::DataCopy(xLocalTmp, xGm[progress + i * this->loop], 32);
                    float tmp = xLocalTmp.GetValue(0);
                    xLocal.SetValue(i, tmp);
                }
                inQueueY.FreeTensor(xLocalTmp);
            }
        }else{
            AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        }
        inQueueX.EnQue<DTYPE_X>(xLocal);
    }

    __aicore__ inline void CopyIn()
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        this->blockSizeAlign32 = (this->blockSize + 32 - 1) / 32 * 32;
        AscendC::DataCopy(xLocal, yGm[0], this->blockSizeAlign32);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        if (this->dataType == 1){
            // return;
            // AscendC::printf("half\n");
            // if (this->dims >= 3 && this->dataSize >= 10000) return;
            // 传入的数据是 half 类型的
            AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
            AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

            // 进行计算需要的临时变量
            AscendC::LocalTensor<float> tempFloat1 = tempQueueFloat.AllocTensor<float>();
            AscendC::LocalTensor<float> tempFloat2 = tempQueueFloat.AllocTensor<float>();
            AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();
            // AscendC::printf("progress = %d\n", progress);

            // 计算
            AscendC::Cast(tempFloat1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            // AscendC::printf("progress = %d\n", progress);
            // 获取tempFloat1中最大的值
            float max = tempFloat1.GetValue(0);
            // AscendC::printf("max = %f\n", max);
            for (int32_t i = 1; i < this->processDataNum; ++i){
                float value = tempFloat1.GetValue(i);
                if (max < value){
                    max = value;
                }
            }
            // AscendC::printf("progress = %d\n", progress);
            // AscendC::ReduceMax(tempFloat2, tempFloat1, workLocal, this->processDataNum);
            // AscendC::printf("progress = %d\n", progress);
            
            // float max = tempFloat2.GetValue(0);
            max = 0 - max;
            // 减去最大值
            AscendC::Adds(tempFloat1, tempFloat1, max, this->processDataNum);
            AscendC::Exp(tempFloat1, tempFloat1, this->processDataNum);
            if (this->processDataNum != this->tileDataNum){
                float tmp = 0;
                for (int32_t i = 0; i < this->ridOfNum; i++){
                    tempFloat1.SetValue(this->processDataNum - i - 1, tmp);
                }
            }
            AscendC::ReduceSum(tempFloat1, tempFloat1, workLocal, this->processDataNum);
            AscendC::Ln(tempFloat1, tempFloat1, 32);
            float tmp = tempFloat1.GetValue(0);
            tempFloat1.SetValue(0, tmp - max);
            AscendC::Cast(yLocal, tempFloat1, AscendC::RoundMode::CAST_NONE, 32);


            // 结果存入输出队列
            outQueueY.EnQue<half>(yLocal);

            // 释放内存
            inQueueX.FreeTensor(xLocal);
            workQueue.FreeTensor(workLocal);
            tempQueueFloat.FreeTensor(tempFloat1);
            tempQueueFloat.FreeTensor(tempFloat2);
        }else if (this->dataType == 0 && this->dims == 2){
            // return;
            // AscendC::printf("float\n");
            // if (this->dims == 3 && this->dim == 2) return;
            // 传入的数据是 float 类型的
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            // AscendC::LocalTensor<uint8_t> dstLocal = dstQueueUint8.AllocTensor<uint8_t>();
            AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();

            // 进行计算需要的临时变量
            // AscendC::LocalTensor<half> tempHalf1 = tempQueueHalf.AllocTensor<half>();
            AscendC::LocalTensor<float> tempFloat1 = tempQueueFloat.AllocTensor<float>();
            // AscendC::LocalTensor<float> tempFloat2 = tempQueueFloat.AllocTensor<float>();

            // 计算
            // for (int i = 0; i < this->processDataNum; ++i){
            //     float t = xLocal.GetValue(i);
            //     AscendC::printf("x[%d] = %f\n", i, t);
            // }
            AscendC::ReduceMax(tempFloat1, xLocal, workLocal, this->processDataNum);
            float max = tempFloat1.GetValue(0);
            max = 0 - max;
            // 减去最大值
            AscendC::Adds(xLocal, xLocal, max, this->processDataNum);
            AscendC::Exp(xLocal, xLocal, this->processDataNum);
            if (this->processDataNum != this->tileDataNum){
                float tmp = 0;
                for (int32_t i = 0; i < this->ridOfNum; i++){
                    xLocal.SetValue(this->processDataNum - i - 1, tmp);
                }
            }
            AscendC::ReduceSum(yLocal, xLocal, workLocal, this->processDataNum);
            float t = yLocal.GetValue(0);
            AscendC::Ln(yLocal, yLocal, 32);
            float tmp = yLocal.GetValue(0);
            yLocal.SetValue(0, tmp - max);
            // AscendC::printf("t = %f, tmp = %f, max = %f\n", t, tmp, max);
            for (int32_t i = 1; i < 32; ++i){
                yLocal.SetValue(i, 0);
            }

            // 结果存入输出队列
            outQueueY.EnQue<float>(yLocal);

            // 释放内存
            inQueueX.FreeTensor(xLocal);
            // tempQueueHalf.FreeTensor(tempHalf1);
            tempQueueFloat.FreeTensor(tempFloat1);
            // tempQueueFloat.FreeTensor(tempFloat2);
            workQueue.FreeTensor(workLocal);
            // dstQueueUint8.FreeTensor(dstLocal);
        }else if (this->dataType == 0 && this->dims == 3){
            // return;
            // AscendC::printf("float = %d\n", this->index);
            // if (this->dim >= 100000) return;
            // 传入的数据是 float 类型的
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            // 进行计算需要的临时变量

            // 计算
            AscendC::Exp(yLocal, xLocal, this->processDataNum);
            if (this->index != 0){
                AscendC::LocalTensor<float> xLocalTmp = inQueueY.DeQue<float>();
                AscendC::Add(yLocal, yLocal, xLocalTmp, this->processDataNum);
                if (this->index == this->loop - 1){
                    AscendC::Ln(yLocal, yLocal, this->processDataNum);
                }
                inQueueY.FreeTensor(xLocalTmp);
            }else{
                if (this->index == this->loop - 1){
                    AscendC::Ln(yLocal, yLocal, this->processDataNum);
                }
            }
            // 结果存入输出队列
            outQueueY.EnQue<float>(yLocal);

            // 释放内存
            inQueueX.FreeTensor(xLocal);
        }else if(this->dataType == 0 && this->dims == 4){
            // return;
            // AscendC::printf("float = %d\n", this->index);
            // if (this->dim >= 0) {
            //     while(true){
            //         AscendC::printf("789\n");
            //     } 
            // }
            // 传入的数据是 float 类型的
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            // 进行计算需要的临时变量

            // 计算
            if (this->index_j == -1){
                if (this->flag == 0){
                    // 求解最大值
                    if (this->index != 0){
                        AscendC::LocalTensor<float> xLocalTmp = inQueueY.DeQue<float>();
                        for (int i = 0; i < this->processDataNum; ++i){
                            float t1 = xLocal.GetValue(i);
                            float t2 = xLocalTmp.GetValue(i);
                            if (t1 < t2){
                                yLocal.SetValue(i, t2);
                            }else{
                                yLocal.SetValue(i, t1);
                            }
                        }
                        inQueueY.FreeTensor(xLocalTmp);
                    }else{
                        for (int i = 0; i < this->processDataNum; ++i){
                            float t = xLocal.GetValue(i);
                            yLocal.SetValue(i, t);
                        }
                    }
                }else{
                    // for (int i = 0; i < this->processDataNum; ++i){
                    //     volatile float t = xLocal.GetValue(i);
                        // volatile float inf = 1.0 / 0;
                        // volatile float negInf = -1.0 / 0;
                    //     if (t == inf){
                    //         while(true){
                    //             AscendC::printf("123\n");
                    //         }
                    //     }
                    //     if (t == negInf){
                    //         while(true){
                    //             AscendC::printf("456\n");
                    //         }
                    //     }
                    // }
                    AscendC::Exp(yLocal, xLocal, this->processDataNum);

                    if (this->index != 0){
                        AscendC::LocalTensor<float> xLocalTmp = inQueueY.DeQue<float>();
                        AscendC::Add(yLocal, yLocal, xLocalTmp, this->processDataNum);
                        if (this->index == this->loop - 1){
                            AscendC::Ln(yLocal, yLocal, this->processDataNum);
                        }
                        // for (int i = 0; i < this->processDataNum; ++i){
                        //     volatile float t1 = xLocal.GetValue(i);
                        //     volatile float t2 = xLocalTmp.GetValue(i);
                        //     if (t1 == inf){
                        //         yLocal.SetValue(i, inf);
                        //     }else if (t1 == negInf){
                        //         yLocal.SetValue(i, negInf);
                        //     }
                        //     if (t2 == inf){
                        //         yLocal.SetValue(i, inf);
                        //     }else if (t2 == negInf){
                        //         yLocal.SetValue(i, negInf);
                        //     }
                        // }
                        inQueueY.FreeTensor(xLocalTmp);
                    }else{
                        if (this->index == this->loop - 1){
                            AscendC::Ln(yLocal, yLocal, this->processDataNum);
                        }
                    }
                }
            }else{
                AscendC::LocalTensor<float> tempFloat1 = tempQueueFloat.AllocTensor<float>();
                AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();

                AscendC::ReduceMax(tempFloat1, xLocal, workLocal, this->dataSize);
                float max = tempFloat1.GetValue(0);
                max = 0 - max;
                // 减去最大值
                AscendC::Adds(yLocal, xLocal, max, this->dataSize);
                AscendC::Exp(yLocal, yLocal, this->dataSize);
                volatile float tmp = 0;
                // AscendC::printf("dataSize = %d, ridOfNum = %d\n", this->dataSize, this->ridOfNum);
                for (int32_t i = 0; i < this->ridOfNum; i++){
                    yLocal.SetValue(this->dataSize - i - 1, tmp);
                }
                AscendC::ReduceSum(yLocal, yLocal, workLocal, this->dataSize);
                AscendC::Ln(yLocal, yLocal, 32);
                tmp = yLocal.GetValue(0) - max;
                yLocal.SetValue(0, tmp);
                // for (int i = 0; i < this->dataSize; ++i){
                //     volatile float t = xLocal.GetValue(i);
                //     if (t != t){
                //         yLocal.SetValue(i, t);
                //         break;
                //     }
                // }
                // volatile float inf = 1.0 / 0;
                // volatile float negInf = -1.0 / 0;
                // for (int i = 0; i < this->processDataNum; ++i){
                //     volatile float t1 = xLocal.GetValue(i);
                //     if (t1 == inf){
                //         yLocal.SetValue(0, inf);
                //         break;
                //     }else if (t1 == negInf){
                //         yLocal.SetValue(0, negInf);
                //         break;
                //     }
                // }

                tempQueueFloat.FreeTensor(tempFloat1);
                workQueue.FreeTensor(workLocal);
            }
            // 结果存入输出队列
            outQueueY.EnQue<float>(yLocal);

            // 释放内存
            inQueueX.FreeTensor(xLocal);
        }

    }

    __aicore__ inline void Compute()
    {
        //考生补充算子计算代码
        if (this->dataType == 1){
            // return;
            // 传入的数据是 half 类型的
            AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
            AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

            // 进行计算需要的临时变量
            AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();
            AscendC::LocalTensor<float> tempFloat1 = tempQueueFloat.AllocTensor<float>();
            AscendC::LocalTensor<float> tempFloat2 = tempQueueFloat.AllocTensor<float>();

            // 计算
            AscendC::Cast(tempFloat1, xLocal, AscendC::RoundMode::CAST_NONE, this->blockSizeAlign32);
            // 获取tempFloat1中最大的值
            AscendC::ReduceMax(tempFloat2, tempFloat1, workLocal, this->blockSizeAlign32);
            float max = tempFloat2.GetValue(0);
            max = 0 - max;
            // 减去最大值
            AscendC::Adds(tempFloat1, tempFloat1, max, this->blockSizeAlign32);
            AscendC::Exp(tempFloat1, tempFloat1, this->blockSizeAlign32);
            for (int32_t i = 0; i < (this->blockSizeAlign32 - this->blockSize); ++i){
                float tmp = 0;
                tempFloat1.SetValue(this->blockSizeAlign32 -1 - i, tmp);
            }
            AscendC::ReduceSum<float>(tempFloat1, tempFloat1, workLocal, this->blockSizeAlign32);
            // float t = tempFloat1.GetValue(0);
            // AscendC::printf("t = %f\n", t);
            // if (t == 65504.0){
            //     half tmp = 1 / 0;
            //     yLocal.SetValue(0, tmp);
            // }
            AscendC::Ln(tempFloat1, tempFloat1, 32);
            float tmp = tempFloat1.GetValue(0);
            tempFloat1.SetValue(0, tmp - max);
            AscendC::Cast(yLocal, tempFloat1, AscendC::RoundMode::CAST_NONE, 32);
            // float t = yLocal.GetValue(0);
            // AscendC::printf("t = %f\n", t);


            // 结果存入输出队列
            outQueueY.EnQue<half>(yLocal);

            // 释放内存
            inQueueX.FreeTensor(xLocal);
            tempQueueFloat.FreeTensor(tempFloat1);
            workQueue.FreeTensor(workLocal);
            tempQueueFloat.FreeTensor(tempFloat2);
        }else{
            // return;
            // 传入的数据是 float 类型的
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            AscendC::LocalTensor<float> workLocal = workQueue.AllocTensor<float>();
            // AscendC::LocalTensor<uint8_t> dstLocal = dstQueueUint8.AllocTensor<uint8_t>();

            // 进行计算需要的临时变量
            // AscendC::LocalTensor<half> tempHalf1 = tempQueueHalf.AllocTensor<half>();
            AscendC::LocalTensor<float> tempFloat1 = tempQueueFloat.AllocTensor<float>();
            // AscendC::LocalTensor<float> tempFloat2 = tempQueueFloat.AllocTensor<float>();

            // 计算
            // 获取tempFloat1中最大的值
            // float max = xLocal.GetValue(0);
            // for (int i = 1; i < this->blockSizeAlign32; ++i){
            //     volatile float value = xLocal.GetValue(i);
            //     if (value != value){
            //     }else{
            //         AscendC::printf("value = %f\n", value);
            //         if (max < value){
            //             max = value;
            //         }
            //     }
            // }
            AscendC::ReduceMax(tempFloat1, xLocal, workLocal, this->blockSizeAlign32);
            float max = tempFloat1.GetValue(0);
            // AscendC::printf("max = %f\n", max);
            max = 0 - max;
            // 减去最大值
            AscendC::Adds(xLocal, xLocal, max, this->blockSizeAlign32);
            AscendC::Exp(xLocal, xLocal, this->blockSizeAlign32);
            for (int32_t i = 0; i < (this->blockSizeAlign32 - this->blockSize); ++i){
                float tmp = 0;
                xLocal.SetValue(this->blockSizeAlign32 -1 - i, tmp);
            }
            AscendC::ReduceSum<float>(xLocal, xLocal, workLocal, this->blockSizeAlign32);
            AscendC::Ln(yLocal, xLocal, 32);
            float tmp = yLocal.GetValue(0);
            yLocal.SetValue(0, tmp - max);

            // 结果存入输出队列
            outQueueY.EnQue<float>(yLocal);

            // 释放内存
            inQueueX.FreeTensor(xLocal);
            // tempQueueHalf.FreeTensor(tempHalf1);
            tempQueueFloat.FreeTensor(tempFloat1);
            // tempQueueFloat.FreeTensor(tempFloat2);
            // dstQueueUint8.FreeTensor(dstLocal);
            workQueue.FreeTensor(workLocal);
        }

    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        if (this->dataType == 0 && this->dims == 3){
            AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        }else if (this->dataType == 0 && this->dims == 4){
            if (this->index_j == -1){
                if (this->flag == 0){
                    // for (int32_t i = 0; i < this->processDataNum; ++i){
                    //     float tmp = yLocal.GetValue(i);
                    //     this->maxGm[progress * this->tileDataNum + i] = tmp;
                    // }
                }else{
                    // if (this->processDataNum == this->tileDataNum){
                        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
                    // }else{
                    //     auto size = this->dataSize % this->tileDataNum;
                    //     for (int32_t i = 0; i < size; ++i){
                    //         float tmp = yLocal.GetValue(i);
                    //         yGm.SetValue(progress * this->tileDataNum + i, tmp);
                    //     }
                    // }
                }
            }else{
                // float tmp = yLocal.GetValue(0);
                // yGm.SetValue(progress, tmp);
                AscendC::DataCopy(yGm[progress], yLocal, 32);
            }
        }else {
            AscendC::DataCopy(yGm[progress], yLocal, 32);
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        float tmp = yLocal.GetValue(0);
        // AscendC::printf("tmp = %f\n", tmp);
        AscendC::DataCopy(yGm[0], yLocal, 32);
        outQueueY.FreeTensor(yLocal);
    }


private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueMax;
    TQue<QuePosition::VECIN, BUFFER_NUM> tempQueueFloat, dstQueueUint8, workQueue;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X> xGm;
    // GlobalTensor<DTYPE_X> maxGm;
    // float * maxGm;
    GlobalTensor<DTYPE_Y> yGm;

    //考生补充自定义成员变量
    uint32_t tileDataNum;
    uint32_t coreDataNum; // 每个核要处理的数据量
    uint32_t coreCarryTimes; // 每个核循环计算的次数
    uint32_t coreFinallDataNum; // 每个核最后处理的数据量
    uint32_t processDataNum; // 每个核每次要处理的数据量
    // DataType type; // 数据类型
    uint32_t dataType; // 运行时数据类型
    int64_t dim;
    bool keepDim;
    uint32_t blockSize;
    uint32_t blockSizeAlign32;
    uint32_t ridOfNum;
    uint32_t dims;
    uint32_t dimSize;
    uint32_t dataSize;
    uint32_t index;
    int32_t index_j;
    uint32_t loop;
    uint32_t loopCp;
    int flag; // 0：求解最大值，1；正常累加
};

extern "C" __global__ __aicore__ void log_sum_exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    //补充init和process函数调用内容
    // float shareArray[524288] = {0};
    uint32_t loop = tiling_data.loop;
    uint32_t loopCp = tiling_data.loopCp;
    uint32_t count = tiling_data.count;
    // AscendC::printf("loop = %d, loopCp = %d, cout = %d, dataSize = %d, ridOfNum = %d\n", loop, loopCp, count, tiling_data.dataSize, tiling_data.ridOfNum);
    uint32_t type = tiling_data.dataType;
    if (tiling_data.dims == 4){
        loop = count;
    }
    for (int32_t i = 0; i < loopCp; ++i){
        KernelLogSumExp op;
        op.Init(x, x, tiling_data.bigCoreDataNum, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreCarryNum, 
            tiling_data.bigCoreFinallDealNum,tiling_data.smallCoreDataNumCp, tiling_data.smallCoreCarryNumCp, tiling_data.smallCoreFinallDealNumCp, 
            tiling_data.dataType, tiling_data.dim, tiling_data.keepDim, tiling_data.blockSizeCp, tiling_data.ridOfNumCp, tiling_data.dims, tiling_data.dimSize,
            tiling_data.dataSizeCp, i, tiling_data.loopCp, -1, 1);
        op.Process();
    }
    for (int32_t i = 0; i < loop; ++i){
        KernelLogSumExp op;
        op.Init(x, y, tiling_data.bigCoreDataNum, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreCarryNum, 
            tiling_data.bigCoreFinallDealNum,tiling_data.smallCoreDataNum, tiling_data.smallCoreCarryNum, tiling_data.smallCoreFinallDealNum, 
            tiling_data.dataType, tiling_data.dim, tiling_data.keepDim, tiling_data.blockSize, tiling_data.ridOfNum, tiling_data.dims, tiling_data.dimSize,
            tiling_data.dataSize, i, tiling_data.loop, 0, -1);
        op.Process();
    }
}