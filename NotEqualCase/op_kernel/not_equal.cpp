#include "kernel_operator.h"
using namespace AscendC;
constexpr int64_t BUFFER_NUM = 2;

class KernelNotEqual {
public:
    __aicore__ inline KernelNotEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t bigCoreDataNum, uint64_t tileDataNum, uint64_t bigCoreNum, uint64_t bigCoreCarryNum, uint64_t bigCoreFinallDealNum,
                                uint64_t smallCoreDataNum, uint64_t smallCoreCarryNum, uint64_t smallCoreFinallDealNum, uint64_t dataType, 
                                uint64_t inputShape00, uint64_t inputShape01, uint64_t inputShape10, uint64_t inputShape11, uint64_t axis, uint64_t who, uint64_t isBroadcast, uint64_t coreNum)
    {
        //考生补充初始化代码
        this->coreNum = coreNum;
        this->inputShape00 = inputShape00;
        this->inputShape01 = inputShape01;
        this->inputShape10 = inputShape10;
        this->inputShape11 = inputShape11;
        this->axis = axis;
        this->who = who;
        this->isBroadcast = isBroadcast;
        this->dataType = dataType;
        // AscendC::printf("dataType: %d\n", this->test);
        uint64_t aicoreIndex = GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * aicoreIndex;
        this->tileDataNum = tileDataNum;
        // this->tileDataNum = 7936; // return fail
        if (aicoreIndex < bigCoreNum){
            // AscendC::printf("bigCore\n");
            this->coreDataNum = bigCoreDataNum;
            this->coreCarryTimes = bigCoreCarryNum;
            this->coreFinallDataNum = bigCoreFinallDealNum;
        }else{
            // AscendC::printf("smallCore\n");
            this->coreDataNum = smallCoreDataNum;
            this->coreCarryTimes = smallCoreCarryNum;
            this->coreFinallDataNum = smallCoreFinallDealNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (aicoreIndex - bigCoreNum);
        }
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX1, 2, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(inQueueX2, 2, this->tileDataNum * sizeof(DTYPE_X2));
        pipe.InitBuffer(outQueueY, 2, this->tileDataNum * sizeof(uint8_t));

        pipe.InitBuffer(inQueueX, 2, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(inQueue_float, 2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(dstQueue, 2, this->tileDataNum);
        // pipe.InitBuffer(inQueue_uint8, 2, this->tileDataNum * sizeof(uint8_t) * 4);
        // pipe.InitBuffer(inQueue_uint64, 2, this->tileDataNum * sizeof(uint64_t));
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        uint64_t loopCount = this->coreCarryTimes;
        this->processDataNum = this->tileDataNum;

        for (int64_t i = 0; i < loopCount; i++) {
            if ( i == this->coreCarryTimes - 1){
                this->processDataNum = this->coreFinallDataNum;
            }
            if (this->processDataNum % 32 != 0) {
                this->processDataNum = (this->processDataNum / 32 + 1) * 32;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();
        // 根据数据量拷贝的大小进行手动广播
        if (this->isBroadcast == 1){
        // if (0){
            if (this->who == 0){
                if (this->axis == 0){
                    // 第一个输入数据需要广播 按照axis = 0的方向进行广播
                    // 计算出第一行的位置
                    auto firstRow = progress * this->tileDataNum / this->inputShape11;
                    // 根据progress算出第一行要加载多少个数据
                    auto first = this->inputShape11 - progress * this->tileDataNum % this->inputShape11;
                    if (first > this->processDataNum){
                        first = this->processDataNum;
                    }
                    auto row = (this->processDataNum - first) / this->inputShape11 + 1; // 除了第一行还需要要拷贝数据多少行
                    auto colLeaf = (this->processDataNum - first) % this->inputShape11;  // 最后一行要拷贝多少数据
                    // 进行第一行的数据拷贝，判断第一行拷贝的起始位置是不是32字节对齐
                    auto whichRow32 = firstRow / 32;
                    auto whichRow32End = firstRow % 32;
                    AscendC::DataCopy(x2Local, x1Gm[whichRow32 * 32], 32);
                    for (int i = 0; i < first; ++i){
                        x1Local.SetValue(i, x2Local.GetValue(whichRow32End));
                    }
                    // 进行中间行和最后一行的数据拷贝
                    for (int i = 0; i < row; i++){
                        auto row32 = firstRow + 1 + i;
                        auto whichRow32 = row32 / 32;
                        auto whichRow32End = row32 % 32;
                        AscendC::DataCopy(x2Local, x1Gm[whichRow32 * 32], 32);
                        if (i == row - 1){
                            for (int j = 0; j < colLeaf; j++){
                                x1Local.SetValue(first + i * this->inputShape11 + j, x2Local.GetValue(whichRow32End));
                            }
                        }else{
                            for (int j = 0; j < this->inputShape11; j++){
                                x1Local.SetValue(first + i * this->inputShape11 + j, x2Local.GetValue(whichRow32End));
                            }
                        }
                    }
                }else{
                    // 第一个输入数据需要广播 按照axis = 1的方向进行广播
                    // 计算第一行要加载多少个数据
                    // AscendC::printf("tileDataNum: %d, progress: %d, processDataNum: %d\n", this->tileDataNum, progress, this->processDataNum);
                    auto first = this->inputShape01 - progress * this->tileDataNum % this->inputShape01;
                    auto size = progress * this->tileDataNum / this->inputShape01 / this->inputShape10 * this->inputShape01;
                    auto row = 0;
                    auto colLeaf = 0;
                    if (first < this->processDataNum){
                        // 表明第一行不够这一次的数据加载
                        row = (this->processDataNum - first) / this->inputShape01 + 1; // 要拷贝数据多少行
                        colLeaf = (this->processDataNum - first) % this->inputShape01;  // 最后一行要拷贝多少数据
                    }else{
                        first = this->processDataNum;
                    }
                    auto whichCol32 = (this->inputShape01 - first) / 32;
                    auto whichCol32End = (this->inputShape01 - first) % 32;
                    auto copy32 = (this->inputShape01 / 32 + 1) * 32; // 拷贝的数据量需要32字节对齐
                    // 加载第一行数据
                    AscendC::DataCopy(x2Local, x1Gm[size + whichCol32 * 32], copy32);
                    for (int i = 0; i < first; i++){
                        x1Local.SetValue(i, x2Local.GetValue(whichCol32End + i));
                    }
                    auto s = progress * this->tileDataNum / this->inputShape01 % this->inputShape10;
                    // AscendC::printf("s: %d\n", s);
                    for (int i = 0; i < row; i++){
                        auto p = (i + 1 + s) / this->inputShape10;
                        auto size_cp = size + p * this->inputShape01;
                        auto copy32 = (this->inputShape01 / 32 + 1) * 32;
                        AscendC::DataCopy(x2Local, x1Gm[size_cp], copy32);
                        if (i == row - 1){
                            for (int j = 0; j < colLeaf; j++){
                                x1Local.SetValue(first + i * this->inputShape01 + j, x2Local.GetValue(j));
                            }
                        }else{
                            for (int j = 0; j < this->inputShape01; j++){
                                x1Local.SetValue(first + i * this->inputShape01 + j, x2Local.GetValue(j));
                            }
                        }
                    }
                }
                AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
            }else{
                if (this->axis == 0){
                    // 第二个输入数据需要广播 按照axis = 0的方向进行广播
                    // 计算第一行的位置
                    auto firstRow = progress * this->tileDataNum / this->inputShape01;
                    // 根据progress算出第一行要加载多少个数据
                    auto first = this->inputShape01 - progress * this->tileDataNum % this->inputShape01;
                    if (first > this->processDataNum){
                        first = this->processDataNum;
                    }
                    auto row = (this->processDataNum - first) / this->inputShape01 + 1; // 要拷贝数据多少行
                    auto colLeaf = (this->processDataNum - first) % this->inputShape01;  // 最后一行要拷贝多少数据
                    // 进行第一行的数据拷贝，判断第一行拷贝的起始位置是不是32字节对齐
                    auto whichRow32 = firstRow / 32;
                    auto whichRow32End = firstRow % 32;
                    AscendC::DataCopy(x1Local, x2Gm[whichRow32 * 32], 32);
                    for (int i = 0; i < first; ++i){
                        x2Local.SetValue(i, x1Local.GetValue(whichRow32End));
                    }
                    for (int i = 0; i < row; i++){
                        auto row32 = firstRow + 1 + i;
                        auto whichRow32 = row32 / 32;
                        auto whichRow32End = row32 % 32;
                        AscendC::DataCopy(x1Local, x2Gm[whichRow32 * 32], 32);
                        if (i == row - 1){
                            for (int j = 0; j < colLeaf; j++){
                                x2Local.SetValue(first + i * this->inputShape01 + j, x1Local.GetValue(whichRow32End));
                            }
                        }else{
                            for (int j = 0; j < this->inputShape01; j++){
                                x2Local.SetValue(first + i * this->inputShape01 + j, x1Local.GetValue(whichRow32End));
                            }
                        }
                    }
                }else{
                    // 第二个输入数据需要广播 按照axis = 1的方向进行广播
                    // 计算第一行要加载多少个数据
                    auto first = this->inputShape11 - progress * this->tileDataNum % this->inputShape11;
                    auto row = 0;
                    auto colLeaf = 0;
                    if (first < this->processDataNum){
                        // 表明第一行不够这一次的数据加载
                        row = (this->processDataNum - first) / this->inputShape11 + 1; // 要拷贝数据多少行
                        colLeaf = (this->processDataNum - first) % this->inputShape11;  // 最后一行要拷贝多少数据
                    }else{
                        first = this->processDataNum;
                    }
                    auto whichCol32 = (this->inputShape11 - first) / 32;
                    auto whichCol32End = (this->inputShape11 - first) % 32;
                    auto copy32 = (first / 32 + 1) * 32; // 拷贝的数据量需要32字节对齐
                    // 加载第一行数据
                    AscendC::DataCopy(x1Local, x2Gm[whichCol32 * 32], copy32);
                    for (int i = 0; i < first; i++){
                        x2Local.SetValue(i, x1Local.GetValue(whichCol32End + i));
                    }
                    // 加载中间行和最后一行的数据
                    for (int i = 0; i < row; i++){
                        auto copy32 = (this->inputShape11 / 32 + 1) * 32;
                        AscendC::DataCopy(x1Local, x2Gm[0], copy32);
                        if (i == row - 1){
                            for (int j = 0; j < colLeaf; j++){
                                x2Local.SetValue(first + i * this->inputShape11 + j, x1Local.GetValue(j));
                            }
                        }else{
                            for (int j = 0; j < this->inputShape11; j++){
                                x2Local.SetValue(first + i * this->inputShape11 + j, x1Local.GetValue(j));
                            }
                        }
                    }
                }
                AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
            }
        }else{
            AscendC::DataCopy(x1Local, x1Gm[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(x2Local, x2Gm[progress * this->tileDataNum], this->processDataNum);
        }
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int64_t progress)
    {
        //考生补充算子计算代码
        // AscendC::printf("Compute\n");
        AscendC::LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();

        // AscendC::LocalTensor<uint64_t> temp7 = inQueue_uint64.AllocTensor<uint64_t>();

        // 根据DTYPE_X1的不同类型进行转换
        if (this->dataType == 2){
            // 传入的数据是int8_t类型的
            // AscendC::printf("int8_t\n");
            AscendC::LocalTensor<int8_t> x1Local_temp = inQueueX1.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> x2Local_temp = inQueueX2.DeQue<int8_t>();

            AscendC::LocalTensor<half> temp1 = inQueueX.AllocTensor<half>();
            AscendC::LocalTensor<half> temp2 = inQueueX.AllocTensor<half>();

            AscendC::Cast(temp1, x1Local_temp, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(temp2, x2Local_temp, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Sub(temp1, temp1, temp2, this->processDataNum);
            AscendC::Abs(temp1, temp1, this->processDataNum);
            half scalar = 1;
            AscendC::Adds(temp2, temp1, scalar, this->processDataNum);
            AscendC::Div(temp1, temp1, temp2, this->processDataNum);
            AscendC::Cast(yLocal, temp1, AscendC::RoundMode::CAST_CEIL, this->processDataNum);

            inQueueX1.FreeTensor(x1Local_temp);
            inQueueX2.FreeTensor(x2Local_temp);
            inQueueX.FreeTensor(temp1);
            inQueueX.FreeTensor(temp2);
        }else if (this->dataType == 3){
            // 传入的数据是int32_t类型的
            // AscendC::printf("int32_t\n");
            AscendC::LocalTensor<int32_t> x1Local_temp = inQueueX1.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> x2Local_temp = inQueueX2.DeQue<int32_t>();

            AscendC::LocalTensor<half> temp1 = inQueueX.AllocTensor<half>();
            AscendC::LocalTensor<float> temp5 = inQueue_float.AllocTensor<float>();
            AscendC::LocalTensor<float> temp6 = inQueue_float.AllocTensor<float>();

            AscendC::Sub(x1Local_temp, x1Local_temp, x2Local_temp, this->processDataNum);
            AscendC::Cast(temp5, x1Local_temp, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
            AscendC::Abs(temp5, temp5, this->processDataNum);
            float scalar = 1;
            AscendC::Adds(temp6, temp5, scalar, this->processDataNum);
            AscendC::Div(temp5, temp5, temp6, this->processDataNum);
            AscendC::Cast(temp1, temp5, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
            AscendC::Cast(yLocal, temp1, AscendC::RoundMode::CAST_CEIL, this->processDataNum);

            inQueueX1.FreeTensor(x1Local_temp);
            inQueueX2.FreeTensor(x2Local_temp);
            inQueueX.FreeTensor(temp1);
            inQueue_float.FreeTensor(temp5);
            inQueue_float.FreeTensor(temp6);
        }else{
            if (this->dataType == 1){
                // if ((this->size0 != this->size1) && (this->dim0 >= 4 || this->dim1 == 1)) return;
                // AscendC::printf("half\n");
                // if (this->coreNum == 1) return;
                
                AscendC::LocalTensor<half> x1Local_temp = inQueueX1.DeQue<half>();
                AscendC::LocalTensor<half> x2Local_temp = inQueueX2.DeQue<half>();

                AscendC::LocalTensor<half> temp1 = inQueueX.AllocTensor<half>();
                AscendC::LocalTensor<float> temp5 = inQueue_float.AllocTensor<float>();
                AscendC::LocalTensor<float> temp6 = inQueue_float.AllocTensor<float>();
                // AscendC::LocalTensor<uint8_t> dst = dstQueue.AllocTensor<uint8_t>();

                AscendC::Cast(temp5, x1Local_temp, AscendC::RoundMode::CAST_NONE, this->processDataNum);
                AscendC::Cast(temp6, x2Local_temp, AscendC::RoundMode::CAST_NONE, this->processDataNum);

                AscendC::Duplicate<half>(temp1, (half)1.0, this->processDataNum);
                AscendC::Compare(yLocal, x1Local_temp, x2Local_temp, CMPMODE::NE, this->processDataNum);
                AscendC::Select(temp1, yLocal, temp1, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
                AscendC::Cast(yLocal, temp1, AscendC::RoundMode::CAST_ROUND , this->processDataNum);
                

                for (int i = 0; i < this->processDataNum; ++i){
                    volatile float t5 = temp5.GetValue(i);
                    volatile float t6 = temp6.GetValue(i);
                    // volatile bool nanDetected = (t5 != t5 || t6 != t6);
                    // AscendC::printf("nanDetected = %d\n", nanDetected);
                    float p1 = 1.0 / 0;
                    float p2 = -1.0 / 0;
                    if (t5 == p1){
                        if (t6 == p2){
                            yLocal.SetValue(i, 1);
                            continue;
                        }
                    }
                    if (t5 == p2){
                        if (t6 == p1){
                            yLocal.SetValue(i, 1);
                            continue;
                        }
                    }
                    if (t5 != t5){
                        // AscendC::printf("t = %f\n", t);
                        yLocal.SetValue(i, 1);
                        // return;
                        continue;
                    }
                    if (t6 != t6){
                        // AscendC::printf("t = %f\n", t);
                        yLocal.SetValue(i, 1);
                        // return;
                        continue;
                    }
                    // if (t5 > t6){
                    //     if ((t5 - t6) < 1e-8){
                    //         yLocal.SetValue(i, 0);
                    //     }else{
                    //         yLocal.SetValue(i, 1);
                    //     }
                    // }else{
                    //     if ((t6 - t5) < 1e-8){
                    //         yLocal.SetValue(i, 0);
                    //     }else{
                    //         yLocal.SetValue(i, 1);
                    //     }
                    // }
                    // if (t5 != t6){
                    //     yLocal.SetValue(i, 1);
                    // }else{
                    //     yLocal.SetValue(i, 0);
                    // }
                }
                inQueueX1.FreeTensor(x1Local_temp);
                inQueueX2.FreeTensor(x2Local_temp);
                inQueueX.FreeTensor(temp1);
                inQueue_float.FreeTensor(temp5);
                inQueue_float.FreeTensor(temp6);
                // dstQueue.FreeTensor(dst);
            }else{
                // AscendC::printf("float\n");
                AscendC::LocalTensor<float> x1Local_temp = inQueueX1.DeQue<float>();
                AscendC::LocalTensor<float> x2Local_temp = inQueueX2.DeQue<float>();

                AscendC::LocalTensor<half> temp1 = inQueueX.AllocTensor<half>();
                AscendC::LocalTensor<float> temp5 = inQueue_float.AllocTensor<float>();
                AscendC::LocalTensor<float> temp6 = inQueue_float.AllocTensor<float>();
                AscendC::LocalTensor<uint8_t> dst = dstQueue.AllocTensor<uint8_t>();

                AscendC::Cast(temp5, x1Local_temp, AscendC::RoundMode::CAST_NONE, this->processDataNum);
                AscendC::Cast(temp6, x2Local_temp, AscendC::RoundMode::CAST_NONE, this->processDataNum);

                AscendC::Duplicate<half>(temp1, (half)1.0, this->processDataNum);
                AscendC::Compare(dst, x1Local_temp, x2Local_temp, CMPMODE::NE, this->processDataNum);
                AscendC::Select(temp1, dst, temp1, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
                AscendC::Cast(yLocal, temp1, AscendC::RoundMode::CAST_NONE , this->processDataNum);

                for (int i = 0; i < this->processDataNum; ++i){
                    volatile float t5 = temp5.GetValue(i);
                    volatile float t6 = temp6.GetValue(i);
                    // volatile bool nanDetected = (t5 != t5 || t6 != t6);
                    // AscendC::printf("nanDetected = %d\n", nanDetected);
                    float p1 = 1.0 / 0;
                    float p2 = -1.0 / 0;
                    if (t5 == p1){
                        if (t6 == p2){
                            yLocal.SetValue(i, 1);
                            continue;
                        }
                    }
                    if (t5 == p2){
                        if (t6 == p1){
                            yLocal.SetValue(i, 1);
                            continue;
                        }
                    }
                    if (t5 != t5){
                        // AscendC::printf("t = %f\n", t);
                        yLocal.SetValue(i, 1);
                        // return;
                        continue;
                    }
                    if (t6 != t6){
                        // AscendC::printf("t = %f\n", t);
                        yLocal.SetValue(i, 1);
                        // return;
                        continue;
                    }
                }
                inQueueX1.FreeTensor(x1Local_temp);
                inQueueX2.FreeTensor(x2Local_temp);
                inQueueX.FreeTensor(temp1);
                inQueue_float.FreeTensor(temp5);
                inQueue_float.FreeTensor(temp6);
                dstQueue.FreeTensor(dst);
            }
        }
        outQueueY.EnQue<uint8_t>(yLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueue_float, dstQueue;
    // TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_half, inQueueX_float, inQueueX_base_half;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X1> x1Gm, x2Gm;
    GlobalTensor<uint8_t> yGm;

    //考生补充自定义成员变量
    uint64_t tileDataNum;
    uint64_t coreDataNum; // 每个核要处理的数据量
    uint64_t coreCarryTimes; // 每个核循环计算的次数
    uint64_t coreFinallDataNum; // 每个核最后处理的数据量
    uint64_t processDataNum; // 每个核每次要处理的数据量
    uint64_t dataType; // 运行时数据类型
    uint64_t compareDistSize;
    uint64_t axis;
    uint64_t who;
    uint64_t isBroadcast;
    uint64_t inputShape00;
    uint64_t inputShape01;
    uint64_t inputShape10;
    uint64_t inputShape11;
    uint64_t coreNum;
};

extern "C" __global__ __aicore__ void not_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    // AscendC::printf("not_equal_start\n");
    KernelNotEqual op;
    op.Init(x1, x2, y, tiling_data.bigCoreDataNum, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreCarryNum, tiling_data.bigCoreFinallDealNum,
        tiling_data.smallCoreDataNum, tiling_data.smallCoreCarryNum, tiling_data.smallCoreFinallDealNum, tiling_data.dataType, 
        tiling_data.inputShape00, tiling_data.inputShape01, tiling_data.inputShape10, tiling_data.inputShape11, tiling_data.axis, 
        tiling_data.who, tiling_data.isBroadcast, tiling_data.coreNum);
    op.Process();
}