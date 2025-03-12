#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelDiv {
public:
    __aicore__ inline KernelDiv() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint32_t tileDataNum, uint32_t smallCoreDataNum, uint32_t smallCoreCarryNum,
        uint32_t smallCoreFinallDealNum, uint32_t bigCoreDataNum, uint32_t bigCoreCarryNum, uint32_t bigCoreFinallDealNum, uint32_t bigCoreNum, uint32_t inputShape00, uint32_t inputShape01, uint32_t inputShape10, uint32_t inputShape11,
        uint8_t dataType, uint8_t dim0, uint8_t dim1, uint8_t axis, uint8_t who, uint8_t isBroadcast)
    {
        // this->type = type;
        //考生补充初始化代码
        this->dim0 = dim0;
        this->dim1 = dim1;
        this->inputShape00 = inputShape00;
        this->inputShape01 = inputShape01;
        this->inputShape10 = inputShape10;
        this->inputShape11 = inputShape11;
        this->axis = axis;
        this->who = who;
        this->isBroadcast = isBroadcast;
        this->dataType = dataType;
        uint32_t aicoreIndex = GetBlockIdx();
        // uint32_t globalBufferIndex = 0;
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
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1 *)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X2 *)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX1, 2, this->tileDataNum * sizeof(DTYPE_X1));
        pipe.InitBuffer(inQueueX2, 2, this->tileDataNum * sizeof(DTYPE_X2));
        pipe.InitBuffer(outQueueY, 2, this->tileDataNum * sizeof(DTYPE_Y));

        pipe.InitBuffer(tempQueueHalf, 2, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tempQueueFloat, 2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tempQueueZero, 2, sizeof(float));
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        uint32_t loopCount = this->coreCarryTimes;
        this->processDataNum = this->tileDataNum;


        for (int32_t i = 0; i < loopCount; i++) {
            if ( i == this->coreCarryTimes - 1){
                this->processDataNum = this->coreFinallDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ bool is_negative_zero(float value) {
        // 将 float 转换为 uint32_t 以检查位表示
        uint32_t bits = *(uint32_t*)&value;

        // 检查符号位是否为1，并且指数和尾数部分都是0
        // 对于负零，只有符号位为1，其他所有位都为0
        return (bits & 0x7FFFFFFF) == 0 && (bits & 0x80000000) != 0;
    }
    __aicore__ bool is_positive_zero(float value) {
        // 将 float 转换为 uint32_t 以检查位表示
        uint32_t bits = *(uint32_t*)&value;
    
        // 对于正零，所有位都应该是0
        return bits == 0;
    }
    __aicore__ bool is_inf_or_neginf(float value) {
        float inf = 1.0 / 0;
        float negInf = -1.0 / 0;
        if (value == inf){
            return true;
        }
        if (value == negInf){
            return true;
        }
        return false;
    }
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_X1> x1Local = inQueueX1.AllocTensor<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X2> x2Local = inQueueX2.AllocTensor<DTYPE_X2>();
        // 根据数据量拷贝的大小进行手动广播
        if (this->isBroadcast == 1){
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
                    // AscendC::printf("tileDataNum = %d, progress = %d, processDataNum = %d\n", this->tileDataNum, progress, this->processDataNum);
                    auto first = this->inputShape01 - progress * this->tileDataNum % this->inputShape01;
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
                    auto copy32 = (first / 32 + 1) * 32; // 拷贝的数据量需要32字节对齐
                    // 加载第一行数据
                    AscendC::DataCopy(x2Local, x1Gm[whichCol32 * 32], copy32);
                    for (int i = 0; i < first; i++){
                        x1Local.SetValue(i, x2Local.GetValue(whichCol32End + i));
                    }
                    copy32 = (this->inputShape01 / 32 + 1) * 32;
                    AscendC::DataCopy(x2Local, x1Gm[0], copy32);
                    for (int i = 0; i < row; i++){
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
                    copy32 = (this->inputShape11 / 32 + 1) * 32;
                    AscendC::DataCopy(x1Local, x2Gm[0], copy32);
                    for (int i = 0; i < row; i++){
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
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        if (this->dataType == 2){
            // 传入的数据是 int8_t 类型的

            // 获取传入的数据以及分配传出数据的空间
            AscendC::LocalTensor<int8_t> x2Local = inQueueX2.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> x1Local = inQueueX1.DeQue<int8_t>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.AllocTensor<int8_t>();

            // 进行计算需要的临时变量
            AscendC::LocalTensor<half> tempHalf1 = tempQueueHalf.AllocTensor<half>();
            AscendC::LocalTensor<half> tempHalf2 = tempQueueHalf.AllocTensor<half>();

            // 数据计算
            AscendC::Cast(tempHalf1, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Cast(tempHalf2, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            AscendC::Div(tempHalf1, tempHalf1, tempHalf2, this->processDataNum);
            AscendC::Cast(yLocal, tempHalf1, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
            
            // 计算结果存入输出队列
            outQueueY.EnQue<int8_t>(yLocal);

            // 释放内存
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
            tempQueueHalf.FreeTensor(tempHalf1);
            tempQueueHalf.FreeTensor(tempHalf2);
        }else if (this->dataType == 3){
            // 传入的数据是 int32_t 类型的
            // 获取传入的数据以及分配传出数据的空间
            AscendC::LocalTensor<int32_t> x1Local = inQueueX1.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> x2Local = inQueueX2.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> yLocal = outQueueY.AllocTensor<int32_t>();

            // 进行计算需要的临时变量
            AscendC::LocalTensor<float> tempFloat1 = tempQueueFloat.AllocTensor<float>();
            AscendC::LocalTensor<float> tempFloat2 = tempQueueFloat.AllocTensor<float>();

            // 数据计算
            AscendC::Cast(tempFloat1, x1Local, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
            AscendC::Cast(tempFloat2, x2Local, AscendC::RoundMode::CAST_CEIL, this->processDataNum);
            AscendC::Div(tempFloat1, tempFloat1, tempFloat2, this->processDataNum);
            AscendC::Cast(yLocal, tempFloat1, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
            
            // 计算结果存入输出队列
            outQueueY.EnQue<int32_t>(yLocal);

            // 释放内存
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
            tempQueueFloat.FreeTensor(tempFloat1);
            tempQueueFloat.FreeTensor(tempFloat2);
        }
        else if (this->dataType == 1){
            // 传入的数据是 half 类型的
            AscendC::LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
            AscendC::LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
            AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

            AscendC::Div(yLocal, x1Local, x2Local, this->processDataNum);
            // 结果存入输出队列
            outQueueY.EnQue<half>(yLocal);

            // 释放内存
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
        else{
            // 传入的数据是 float 类型的
            AscendC::LocalTensor<float> x1Local = inQueueX1.DeQue<float>();
            AscendC::LocalTensor<float> x2Local = inQueueX2.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

            // 计算
            AscendC::Div(yLocal, x1Local, x2Local,  this->processDataNum);

            // 结果存入输出队列
            outQueueY.EnQue<float>(yLocal);

            // 释放内存
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }

    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    TQue<QuePosition::VECIN, BUFFER_NUM> tempQueueHalf, tempQueueFloat, tempQueueZero;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X1> x1Gm, x2Gm;
    GlobalTensor<DTYPE_Y> yGm;

    //考生补充自定义成员变量
    uint32_t tileDataNum;
    uint32_t coreDataNum; // 每个核要处理的数据量
    uint32_t coreCarryTimes; // 每个核循环计算的次数
    uint32_t coreFinallDataNum; // 每个核最后处理的数据量
    uint32_t processDataNum; // 每个核每次要处理的数据量
    // DataType type; // 数据类型
    uint8_t dataType; // 运行时数据类型
    // uint32_t size1;
    // uint32_t size2;
    uint32_t inputShape00;
    uint32_t inputShape01;
    uint32_t inputShape10;
    uint32_t inputShape11;
    uint8_t dim0;
    uint8_t dim1;
    uint8_t axis;
    uint8_t who;
    uint8_t isBroadcast;

};


extern "C" __global__ __aicore__ void div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    // TODO: user kernel impl
    GET_TILING_DATA(tiling_data, tiling);
    KernelDiv op;
    //补充init和process函数调用内容
    op.Init(x1, x2, y, tiling_data.tileDataNum, tiling_data.smallCoreDataNum, tiling_data.smallCoreCarryNum, tiling_data.smallCoreFinallDealNum, tiling_data.bigCoreDataNum, tiling_data.bigCoreCarryNum, tiling_data.bigCoreFinallDealNum, tiling_data.bigCoreNum,
        tiling_data.inputShape00, tiling_data.inputShape01, tiling_data.inputShape10, tiling_data.inputShape11, tiling_data.dataType, tiling_data.dim0,tiling_data.dim1, tiling_data.axis, 
            tiling_data.who, tiling_data.isBroadcast);
    op.Process();
}
