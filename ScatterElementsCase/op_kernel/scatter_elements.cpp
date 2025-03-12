#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelScatterElements {
public:
    __aicore__ inline KernelScatterElements() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, uint32_t bigCoreDataNum, uint32_t tileDataNum, uint32_t bigCoreNum, uint32_t bigCoreCarryNum, uint32_t bigCoreFinallDealNum, uint32_t smallCoreDataNum, uint32_t smallCoreCarryNum, uint32_t smallCoreFinallDealNum, uint32_t dataType, uint32_t axis, uint32_t reduce, uint32_t dims, uint32_t ridOfNum, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t varDim0, uint32_t varDim1, uint32_t varDim2, uint32_t varDim3, uint32_t interval)
    {
        this->interval = interval;
        this->ridOfNum = ridOfNum;
        this->dim0 = dim0;
        this->dim1 = dim1;
        this->dim2 = dim2;
        this->dim3 = dim3;
        this->varDim0 = varDim0;
        this->varDim1 = varDim1;
        this->varDim2 = varDim2;
        this->varDim3 = varDim3;
        this->axis = axis;
        this->reduce = reduce;
        this->dims = dims;
        this->dataType = dataType;
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
        if (this->dataType == 1){
            varGm_half.SetGlobalBuffer((__gm__ half *)var + globalBufferIndex, this->coreDataNum);
            indicesGm_int32.SetGlobalBuffer((__gm__ int32_t *)indices + globalBufferIndex, this->coreDataNum);
            updatesGm_half.SetGlobalBuffer((__gm__ half *)updates + globalBufferIndex, this->coreDataNum);

            pipe.InitBuffer(inQueueIndices, BUFFER_NUM, this->tileDataNum * sizeof(int32_t));
            pipe.InitBuffer(inQueueUpdates, BUFFER_NUM, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(inQueue_half, BUFFER_NUM, 32 * sizeof(half));
        }else if(this->dataType == 0){
            varGm_float.SetGlobalBuffer((__gm__ float *)var + globalBufferIndex, this->coreDataNum);
            indicesGm_int32.SetGlobalBuffer((__gm__ int32_t *)indices + globalBufferIndex, this->coreDataNum);
            updatesGm_float.SetGlobalBuffer((__gm__ float *)updates + globalBufferIndex, this->coreDataNum);

            pipe.InitBuffer(inQueueIndices, BUFFER_NUM, this->tileDataNum * sizeof(int32_t));
            pipe.InitBuffer(inQueueUpdates, BUFFER_NUM, this->tileDataNum * sizeof(float));
        }else if (this->dataType == 3){
            varGm_int32.SetGlobalBuffer((__gm__ int32_t *)var + globalBufferIndex, this->coreDataNum);
            indicesGm_int32.SetGlobalBuffer((__gm__ int32_t *)indices + globalBufferIndex, this->coreDataNum);
            updatesGm_int32.SetGlobalBuffer((__gm__ int32_t *)updates + globalBufferIndex, this->coreDataNum);

            pipe.InitBuffer(inQueueIndices, BUFFER_NUM, this->tileDataNum * sizeof(int32_t));
            pipe.InitBuffer(inQueueUpdates, BUFFER_NUM, this->tileDataNum * sizeof(int32_t));
            pipe.InitBuffer(intQueue_int64, BUFFER_NUM, this->tileDataNum * sizeof(int64_t));
            pipe.InitBuffer(intQueue_int32, BUFFER_NUM, this->tileDataNum * sizeof(int32_t));
        }else if (this->dataType == 4){
            varGm_uint8.SetGlobalBuffer((__gm__ uint8_t *)var + globalBufferIndex, this->coreDataNum);
            indicesGm_int32.SetGlobalBuffer((__gm__ int32_t *)indices + globalBufferIndex, this->coreDataNum);
            updatesGm_uint8.SetGlobalBuffer((__gm__ uint8_t *)updates + globalBufferIndex, this->coreDataNum);

            pipe.InitBuffer(inQueueIndices, BUFFER_NUM, this->tileDataNum * sizeof(int32_t));
            pipe.InitBuffer(inQueueUpdates, BUFFER_NUM, this->tileDataNum * sizeof(uint8_t));
        }

    }
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->coreCarryTimes;
        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount; i++) {
            if ( i == this->coreCarryTimes - 1){
                this->processDataNum = this->coreFinallDataNum;
            }
            // if (this->processDataNum % 32 != 0) {
            //     this->processDataNum = (this->processDataNum / 32 + 1) * 32;
            // }
            CopyIn(i);
            Compute(i);
            // CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        if (this->dataType == 1){
            // half
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
            AscendC::LocalTensor<half> updatesLocal = inQueueUpdates.AllocTensor<half>();
            AscendC::DataCopy(indicesLocal, indicesGm_int32[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(updatesLocal, updatesGm_half[progress * this->tileDataNum], this->processDataNum);
            inQueueIndices.EnQue(indicesLocal);
            inQueueUpdates.EnQue(updatesLocal);
        }else if (this->dataType == 0){
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
            AscendC::LocalTensor<float> updatesLocal = inQueueUpdates.AllocTensor<float>();
            AscendC::DataCopy(indicesLocal, indicesGm_int32[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(updatesLocal, updatesGm_float[progress * this->tileDataNum], this->processDataNum);
            inQueueIndices.EnQue(indicesLocal);
            inQueueUpdates.EnQue(updatesLocal);
        }else if (this->dataType == 3){
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> updatesLocal = inQueueUpdates.AllocTensor<int32_t>();
            AscendC::DataCopy(indicesLocal, indicesGm_int32[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(updatesLocal, updatesGm_int32[progress * this->tileDataNum], this->processDataNum);
            inQueueIndices.EnQue(indicesLocal);
            inQueueUpdates.EnQue(updatesLocal);
        }else if(this->dataType == 4){
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
            AscendC::LocalTensor<uint8_t> updatesLocal = inQueueUpdates.AllocTensor<uint8_t>();
            AscendC::DataCopy(indicesLocal, indicesGm_int32[progress * this->tileDataNum], this->processDataNum);
            AscendC::DataCopy(updatesLocal, updatesGm_uint8[progress * this->tileDataNum], this->processDataNum);
            inQueueIndices.EnQue(indicesLocal);
            inQueueUpdates.EnQue(updatesLocal);
        }
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        if (this->dataType == 0){
            // float
            if (this->dims == 2){
                AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.DeQue<int32_t>();
                AscendC::LocalTensor<float> updatesLocal = inQueueUpdates.DeQue<float>();
                uint32_t size = this->processDataNum;
                if (this->processDataNum != this->tileDataNum){
                    size = this->processDataNum - this->ridOfNum;
                }
                for (int i = 0; i < size; i++){
                    uint32_t old_index = (i + progress * this->tileDataNum) % this->dim1;
                    int32_t index = indicesLocal.GetValue(i);
                    index = index * this->interval + old_index;
                    float value = updatesLocal.GetValue(i);
                    this->varGm_float.SetValue(index, value);
                }
                inQueueIndices.FreeTensor(indicesLocal);
                inQueueUpdates.FreeTensor(updatesLocal);
            }else{
                AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.DeQue<int32_t>();
                AscendC::LocalTensor<float> updatesLocal = inQueueUpdates.DeQue<float>();
                uint32_t size = this->processDataNum;
                if (this->processDataNum != this->tileDataNum){
                    size = this->processDataNum - this->ridOfNum;
                }
                for (int i = 0; i < size; i++){
                    uint32_t old = i + progress * this->tileDataNum;
                    old = old % (this->dim1 * this->dim2);
                    uint32_t index1 = old / this->dim2;
                    uint32_t index2 = old % this->dim2;
                    auto index = indicesLocal.GetValue(i);
                    index = index * this->interval + index1 * this->varDim2 + index2;
                    float value = updatesLocal.GetValue(i);
                    this->varGm_float.SetValue(index, value);
                }
                inQueueIndices.FreeTensor(indicesLocal);
                inQueueUpdates.FreeTensor(updatesLocal);
            }
        }else if (this->dataType == 1){
            // half
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.DeQue<int32_t>();
            AscendC::LocalTensor<half> updatesLocal = inQueueUpdates.DeQue<half>();

            AscendC::LocalTensor<half> half1 = inQueue_half.AllocTensor<half>();
            AscendC::LocalTensor<half> half2 = inQueue_half.AllocTensor<half>();

            uint32_t size = this->processDataNum;
            if (this->processDataNum != this->tileDataNum){
                size = this->processDataNum - this->ridOfNum;
            }
            for (int i = 0; i < this->processDataNum; i++){
                int32_t index = indicesLocal.GetValue(i);
                half value = updatesLocal.GetValue(i);
                half old = this->varGm_half.GetValue(index);
                half1.SetValue(0, value);
                half2.SetValue(0, old);
                AscendC::Add(half1, half2, half1, 32);
                value = half1.GetValue(0);
                this->varGm_half.SetValue(index, value);
            }
            inQueueIndices.FreeTensor(indicesLocal);
            inQueueUpdates.FreeTensor(updatesLocal);
            inQueue_half.FreeTensor(half1);
            inQueue_half.FreeTensor(half2);
        }else if (this->dataType == 3){
            // int32
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> updatesLocal = inQueueUpdates.DeQue<int32_t>();

            AscendC::LocalTensor<int64_t> int64Local1 = intQueue_int64.AllocTensor<int64_t>();
            AscendC::LocalTensor<int64_t> int64Local2 = intQueue_int64.AllocTensor<int64_t>();
            AscendC::LocalTensor<int32_t> int32Local = intQueue_int32.AllocTensor<int32_t>();

            AscendC::Cast(int64Local1, updatesLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            uint32_t size = this->processDataNum;
            if (this->processDataNum != this->tileDataNum){
                size = this->processDataNum - this->ridOfNum;
            }
            int64_t max = 2147483647;
            int64_t min = -2147483648;
            int64_t range = max - min + 1;
            for (int i = 0; i < size; ++i){
                uint32_t old = i + progress * this->tileDataNum;
                uint32_t index0 = old / (this->dim1 * this->dim2);
                old = old % (this->dim1 * this->dim2);
                uint32_t index1 = old / this->dim2;
                uint32_t index2 = old % this->dim2;
                int32_t index = indicesLocal.GetValue(i);
                if (this->axis == 0){
                    index = index * this->varDim1 * this->varDim2 + index1 * this->varDim2 + index2;
                }else if (this->axis == 1){
                    index = index0 * this->varDim1 * this->varDim2 + index * this->varDim2 + index2;
                }else{
                    index = index0 * this->varDim1 * this->varDim2 + index1 * this->varDim2 + index;
                }
                // index = index0 * this->varDim1 * this->varDim2 + index * this->varDim2 + index2;
                int64_t new_value = int64Local1.GetValue(i);
                int64Local1.SetValue(0, new_value);
                int32_t oldp = this->varGm_int32.GetValue(index);
                int32Local.SetValue(0, oldp);
                AscendC::Cast(int64Local2, int32Local, AscendC::RoundMode::CAST_NONE, 32);
                AscendC::Mul(int64Local1, int64Local2, int64Local1, 32);
                int64_t value = int64Local1.GetValue(0);
                if (value > max){
                    value = (value - min) % range + min;
                }
                if (value < min){
                    value = (value - min) % range + min;
                }
                AscendC::Cast(int32Local, int64Local1, AscendC::RoundMode::CAST_NONE, 32);
                int32_t value32 = int32Local.GetValue(0);
                this->varGm_int32.SetValue(index, value);
            }
            inQueueIndices.FreeTensor(indicesLocal);
            inQueueUpdates.FreeTensor(updatesLocal);
            intQueue_int64.FreeTensor(int64Local1);
            intQueue_int64.FreeTensor(int64Local2);
            intQueue_int32.FreeTensor(int32Local);
        }else if (this->dataType == 4){
            // uint8
            AscendC::LocalTensor<int32_t> indicesLocal = inQueueIndices.DeQue<int32_t>();
            AscendC::LocalTensor<uint8_t> updatesLocal = inQueueUpdates.DeQue<uint8_t>();

            uint32_t size = this->processDataNum;
            if (this->processDataNum != this->tileDataNum){
                size = this->processDataNum - this->ridOfNum;
            }
            for (int i = 0; i < size; i++){
                uint32_t old = i + progress * this->tileDataNum;
                uint32_t index0 = old / (this->dim1 * this->dim2 * this->dim3);
                old = old % (this->dim1 * this->dim2 * this->dim3);
                uint32_t index1 = old / (this->dim2 * this->dim3);
                old = old % (this->dim2 * this->dim3);
                uint32_t index2 = old / this->dim3;
                uint32_t index3 = old % this->dim3;
                auto index = indicesLocal.GetValue(i);
                index = index0 * this->varDim1 * this->varDim2 * this->varDim3 + index1 * this->varDim2 * this->varDim3 + index * this->varDim3 + index3;
                auto old_value = this->varGm_uint8.GetValue(index);
                uint8_t value = updatesLocal.GetValue(i);
                value += old_value;
                this->varGm_uint8.SetValue(index, value);
            }
            inQueueIndices.FreeTensor(indicesLocal);
            inQueueUpdates.FreeTensor(updatesLocal);
        }
        
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // AscendC::LocalTensor<uint8_t> yLocal = outQueueY.DeQue<uint8_t>();
        // AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        // outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueVar, inQueueIndices, inQueueUpdates;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_half, intQueue_int64, intQueue_int32;
    //create queue for output, in this case depth is equal to buffer num
    // TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> varGm_half, updatesGm_half;
    GlobalTensor<float> varGm_float, updatesGm_float;
    GlobalTensor<int32_t> varGm_int32;
    GlobalTensor<int32_t> updatesGm_int32;
    GlobalTensor<uint8_t> varGm_uint8, updatesGm_uint8;
    GlobalTensor<int32_t> indicesGm_int32;

    //考生补充自定义成员变量
    uint32_t tileDataNum;
    uint32_t coreDataNum; // 每个核要处理的数据量
    uint32_t coreCarryTimes; // 每个核循环计算的次数
    uint32_t coreFinallDataNum; // 每个核最后处理的数据量
    uint32_t processDataNum; // 每个核每次要处理的数据量
    uint32_t dataType; // 运行时数据类型
    uint32_t dims; // 维度信息
    uint32_t reduce;
    uint32_t axis;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t dim3;
    uint32_t varDim0;
    uint32_t varDim1;
    uint32_t varDim2;
    uint32_t varDim3;
    uint32_t ridOfNum;
    uint32_t interval;
};

extern "C" __global__ __aicore__ void scatter_elements(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelScatterElements op;
    op.Init(var, indices, updates, tiling_data.bigCoreDataNum, tiling_data.tileDataNum, tiling_data.bigCoreNum, tiling_data.bigCoreCarryNum, tiling_data.bigCoreFinallDealNum,tiling_data.smallCoreDataNum, tiling_data.smallCoreCarryNum, tiling_data.smallCoreFinallDealNum, tiling_data.dataType, tiling_data.axis, tiling_data.reduce, tiling_data.dims, tiling_data.ridOfNum, tiling_data.dim0, tiling_data.dim1, tiling_data.dim2, tiling_data.dim3, tiling_data.varDim0, tiling_data.varDim1, tiling_data.varDim2, tiling_data.varDim3, tiling_data.interval);
    op.Process();
}