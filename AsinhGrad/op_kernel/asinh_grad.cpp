#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAsinhGrad{

public:
    __aicore__ inline KernelAsinhGrad(){}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z,uint32_t daType, uint32_t totalLength,uint32_t blockLength,uint32_t tileNum,uint32_t tileLength,uint32_t lastTileLength)
    {
        // 初始化
        this->daType = daType;
        this->totalLength = totalLength;
        AscendC::printf("totalLength:%d\n",totalLength);
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        AscendC::printf("tileNum:%d\n",tileNum);
        this->tileLength = tileLength / BUFFER_NUM;
        AscendC::printf("tileLength:%d\n",this->tileLength);
        this->lastTileLength = lastTileLength;

        // 初始化管道，这里用了doubleBUffer
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        dyGm.SetGlobalBuffer((__gm__ DTYPE_DY *)dy + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * GetBlockIdx(), 
        this->blockLength);


        // inQueueTmpX
        if(this->daType == 2)
        {
            pipe.InitBuffer(inQueueHighTmpY, 9, this->tileLength * sizeof(float));
        }
        else if(this->daType == 4)
        {
            pipe.InitBuffer(inQueueTmpY, 7, this->tileLength * sizeof(DTYPE_Y));
        }
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_DY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        AscendC::printf("loopCount %d\n",loopCount);
        AscendC::printf("lastTileLength %d\n",this->lastTileLength);
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i,this->tileLength);
            Compute(i,this->tileLength);
            CopyOut(i,this->tileLength);
        }
        if(this->lastTileLength!=0)
        {
            CopyIn(loopCount,this->lastTileLength);
            Compute(loopCount,this->lastTileLength);
            CopyOut(loopCount,this->lastTileLength);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress,uint32_t real_tileLength)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_DY> dyLocal = inQueueY.AllocTensor<DTYPE_DY>();
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], real_tileLength);
        AscendC::DataCopy(dyLocal, dyGm[progress * this->tileLength], real_tileLength);
        inQueueY.EnQue(yLocal);
        inQueueDY.EnQue(dyLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress,uint32_t real_tileLength)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, real_tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
    __aicore__ inline void Compute1(int32_t progress,uint32_t real_tileLength)
    {
        if(this->daType == 4)
        {
            // 阈值：44，小于90，精细计算，大于44，近似计算，
            LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
            LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
            LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
            LocalTensor<DTYPE_Y> yTmpLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();  
            LocalTensor<DTYPE_Y> ySmallSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>(); 
            LocalTensor<DTYPE_Y> ySmallSignSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>(); 
            LocalTensor<DTYPE_Y> yMidSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yMidSignSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yBigSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            // LocalTensor<DTYPE_Y> yBigSignSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            

            AscendC::Abs(yLocal,yLocal,real_tileLength);
            DTYPE_Y not_value = -1;
            // 获取大于90的数
            DTYPE_Y big_scalar = -90;
            AscendC::Adds(yBigSecLocal,yLocal,big_scalar,real_tileLength);
            AscendC::Relu(yBigSecLocal,yBigSecLocal,real_tileLength);
            DTYPE_Y small_value = 1e-38;
            AscendC::Adds(yTmpLocal,yBigSecLocal,small_value,real_tileLength);
            AscendC::Div(yBigSecLocal,yBigSecLocal,yTmpLocal,real_tileLength);
            AscendC::Mul(yBigSecLocal,yLocal,yBigSecLocal,real_tileLength);
            // 获取大于44小于90的数
            DTYPE_Y mid_scalar = -45;
            AscendC::Adds(yMidSecLocal,yLocal,mid_scalar,real_tileLength);
            AscendC::Relu(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yLocal,yMidSignSecLocal,real_tileLength);
            // 获取小于44的数
            AscendC::Sub(ySmallSecLocal,yLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(ySmallSignSecLocal,ySmallSecLocal,small_value,real_tileLength);
            AscendC::Div(ySmallSignSecLocal,ySmallSecLocal,ySmallSignSecLocal,real_tileLength);
            // 获取大于44小于90的数
            AscendC::Sub(yMidSecLocal,yMidSecLocal,yBigSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            
            // 处理小于44的数
            AscendC::Exp(yTmpLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(yTmpLocal,yTmpLocal,yTmpLocal,real_tileLength);
            DTYPE_Y add_scalar = 1;
            AscendC::Adds(yTmpLocal,yTmpLocal,add_scalar,real_tileLength);
            AscendC::Ln(yTmpLocal,yTmpLocal,real_tileLength);
            AscendC::Sub(ySmallSecLocal,ySmallSecLocal,yTmpLocal,real_tileLength);
            AscendC::Exp(ySmallSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(ySmallSecLocal,ySmallSignSecLocal,ySmallSecLocal,real_tileLength);
            // 处理大于45小于90的数
            AscendC::Exp(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Reciprocal(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            // 两者相加
            AscendC::Add(yLocal,yMidSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(zLocal,yLocal,dyLocal,real_tileLength);
            DTYPE_Y mul_scalar = 2;
            AscendC::Muls(zLocal,zLocal,mul_scalar,real_tileLength);
            outQueueZ.EnQue<DTYPE_Z>(zLocal);
            inQueueY.FreeTensor(yLocal);
            inQueueDY.FreeTensor(dyLocal);
            inQueueTmpY.FreeTensor(yTmpLocal);
            inQueueTmpY.FreeTensor(ySmallSecLocal);
            inQueueTmpY.FreeTensor(ySmallSignSecLocal);
            inQueueTmpY.FreeTensor(yMidSecLocal);
            inQueueTmpY.FreeTensor(yMidSignSecLocal);
            inQueueTmpY.FreeTensor(yBigSecLocal);
            inQueueTmpY.FreeTensor(yBigSignSecLocal);
        }
        else if(this->daType == 2)
        {
            LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
            LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
            LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

            LocalTensor<float> yHighLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> dyHighLocal = inQueueHighTmpY.AllocTensor<float>();
            AscendC::Cast(yHighLocal,yLocal,RoundMode::CAST_NONE,real_tileLength);
            AscendC::Cast(dyHighLocal,dyLocal,RoundMode::CAST_NONE,real_tileLength);

            LocalTensor<float> yTmpLocal = inQueueHighTmpY.AllocTensor<float>();  
            LocalTensor<float> ySmallSecLocal = inQueueHighTmpY.AllocTensor<float>(); 
            LocalTensor<float> ySmallSignSecLocal = inQueueHighTmpY.AllocTensor<float>(); 
            LocalTensor<float> yMidSecLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> yMidSignSecLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> yBigSecLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> yBigSignSecLocal = inQueueHighTmpY.AllocTensor<float>();


            AscendC::Abs(yHighLocal,yHighLocal,real_tileLength);
            float not_value = -1;
            // 获取大于90的数
            float big_scalar = -90;
            AscendC::Adds(yBigSecLocal,yHighLocal,big_scalar,real_tileLength);
            AscendC::Relu(yBigSecLocal,yBigSecLocal,real_tileLength);
            float small_value = 1e-38;
            AscendC::Adds(yBigSignSecLocal,yBigSecLocal,small_value,real_tileLength);
            AscendC::Div(yBigSecLocal,yBigSecLocal,yBigSignSecLocal,real_tileLength);
            AscendC::Mul(yBigSecLocal,yHighLocal,yBigSecLocal,real_tileLength);
            // 获取大于44小于90的数
            float mid_scalar = -45;
            AscendC::Adds(yMidSecLocal,yHighLocal,mid_scalar,real_tileLength);
            AscendC::Relu(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yHighLocal,yMidSignSecLocal,real_tileLength);
            // 获取小于44的数
            AscendC::Sub(ySmallSecLocal,yHighLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(ySmallSignSecLocal,ySmallSecLocal,small_value,real_tileLength);
            AscendC::Div(ySmallSignSecLocal,ySmallSecLocal,ySmallSignSecLocal,real_tileLength);
            // 获取大于44小于90的数
            AscendC::Sub(yMidSecLocal,yMidSecLocal,yBigSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            
            // 处理小于44的数
            AscendC::Exp(yTmpLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(yTmpLocal,yTmpLocal,yTmpLocal,real_tileLength);
            float add_scalar = 1;
            AscendC::Adds(yTmpLocal,yTmpLocal,add_scalar,real_tileLength);
            AscendC::Ln(yTmpLocal,yTmpLocal,real_tileLength);
            AscendC::Sub(ySmallSecLocal,ySmallSecLocal,yTmpLocal,real_tileLength);
            AscendC::Exp(ySmallSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(ySmallSecLocal,ySmallSignSecLocal,ySmallSecLocal,real_tileLength);
            // 处理大于45小于90的数
            AscendC::Exp(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Reciprocal(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            // 两者相加
            AscendC::Add(yHighLocal,yMidSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(yHighLocal,yHighLocal,dyHighLocal,real_tileLength);
            float mul_scalar = 2;
            AscendC::Muls(yHighLocal,yHighLocal,mul_scalar,real_tileLength);
            AscendC::Cast(zLocal,yHighLocal,RoundMode::CAST_NONE,real_tileLength);
            outQueueZ.EnQue<DTYPE_Z>(zLocal);
            inQueueY.FreeTensor(yLocal);
            inQueueDY.FreeTensor(dyLocal);
            inQueueHighTmpY.FreeTensor(yHighLocal);
            inQueueHighTmpY.FreeTensor(dyHighLocal);
            inQueueHighTmpY.FreeTensor(yTmpLocal);
            inQueueHighTmpY.FreeTensor(ySmallSecLocal);
            inQueueHighTmpY.FreeTensor(ySmallSignSecLocal);
            inQueueHighTmpY.FreeTensor(yMidSecLocal);
            inQueueHighTmpY.FreeTensor(yMidSignSecLocal);
            inQueueHighTmpY.FreeTensor(yBigSecLocal);
            inQueueHighTmpY.FreeTensor(yBigSignSecLocal);
        }
    }

    __aicore__ inline void Compute(int32_t progress,uint32_t real_tileLength)
    {
        if(this->daType == 4)
        {
            // 阈值：44，小于90，精细计算，大于44，近似计算，
            LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
            LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
            LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
            LocalTensor<DTYPE_Y> yTmpLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();  
            LocalTensor<DTYPE_Y> ySmallSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>(); 
            LocalTensor<DTYPE_Y> ySmallSignSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>(); 
            LocalTensor<DTYPE_Y> yMidSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yMidSignSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yBigSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_Y> yBigSignSecLocal = inQueueTmpY.AllocTensor<DTYPE_Y>();
            

            AscendC::Abs(yLocal,yLocal,real_tileLength);
            DTYPE_Y not_value = -1;
            // 获取大于90的数
            DTYPE_Y big_scalar = -90;
            AscendC::Adds(yBigSecLocal,yLocal,big_scalar,real_tileLength);
            AscendC::Relu(yBigSecLocal,yBigSecLocal,real_tileLength);
            DTYPE_Y small_value = 1e-38;
            AscendC::Adds(yBigSignSecLocal,yBigSecLocal,small_value,real_tileLength);
            AscendC::Div(yBigSecLocal,yBigSecLocal,yBigSignSecLocal,real_tileLength);
            AscendC::Mul(yBigSecLocal,yLocal,yBigSecLocal,real_tileLength);
            // 获取大于44小于90的数
            DTYPE_Y mid_scalar = -45;
            AscendC::Adds(yMidSecLocal,yLocal,mid_scalar,real_tileLength);
            AscendC::Relu(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yLocal,yMidSignSecLocal,real_tileLength);
            // 获取小于44的数
            AscendC::Sub(ySmallSecLocal,yLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(ySmallSignSecLocal,ySmallSecLocal,small_value,real_tileLength);
            AscendC::Div(ySmallSignSecLocal,ySmallSecLocal,ySmallSignSecLocal,real_tileLength);
            // 获取大于44小于90的数
            AscendC::Sub(yMidSecLocal,yMidSecLocal,yBigSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            
            // 处理小于44的数
            AscendC::Exp(yTmpLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(yTmpLocal,yTmpLocal,yTmpLocal,real_tileLength);
            DTYPE_Y add_scalar = 1;
            AscendC::Adds(yTmpLocal,yTmpLocal,add_scalar,real_tileLength);
            AscendC::Ln(yTmpLocal,yTmpLocal,real_tileLength);
            AscendC::Sub(ySmallSecLocal,ySmallSecLocal,yTmpLocal,real_tileLength);
            AscendC::Exp(ySmallSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(ySmallSecLocal,ySmallSignSecLocal,ySmallSecLocal,real_tileLength);
            // 处理大于45小于90的数
            AscendC::Exp(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Reciprocal(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            // 两者相加
            AscendC::Add(yLocal,yMidSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(zLocal,yLocal,dyLocal,real_tileLength);
            DTYPE_Y mul_scalar = 2;
            AscendC::Muls(zLocal,zLocal,mul_scalar,real_tileLength);
            outQueueZ.EnQue<DTYPE_Z>(zLocal);
            inQueueY.FreeTensor(yLocal);
            inQueueDY.FreeTensor(dyLocal);
            inQueueTmpY.FreeTensor(yTmpLocal);
            inQueueTmpY.FreeTensor(ySmallSecLocal);
            inQueueTmpY.FreeTensor(ySmallSignSecLocal);
            inQueueTmpY.FreeTensor(yMidSecLocal);
            inQueueTmpY.FreeTensor(yMidSignSecLocal);
            inQueueTmpY.FreeTensor(yBigSecLocal);
            inQueueTmpY.FreeTensor(yBigSignSecLocal);
        }
        else if(this->daType == 2)
        {
            LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
            LocalTensor<DTYPE_DY> dyLocal = inQueueDY.DeQue<DTYPE_DY>();
            LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();

            LocalTensor<float> yHighLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> dyHighLocal = inQueueHighTmpY.AllocTensor<float>();
            AscendC::Cast(yHighLocal,yLocal,RoundMode::CAST_NONE,real_tileLength);
            AscendC::Cast(dyHighLocal,dyLocal,RoundMode::CAST_NONE,real_tileLength);

            LocalTensor<float> yTmpLocal = inQueueHighTmpY.AllocTensor<float>();  
            LocalTensor<float> ySmallSecLocal = inQueueHighTmpY.AllocTensor<float>(); 
            LocalTensor<float> ySmallSignSecLocal = inQueueHighTmpY.AllocTensor<float>(); 
            LocalTensor<float> yMidSecLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> yMidSignSecLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> yBigSecLocal = inQueueHighTmpY.AllocTensor<float>();
            LocalTensor<float> yBigSignSecLocal = inQueueHighTmpY.AllocTensor<float>();


            AscendC::Abs(yHighLocal,yHighLocal,real_tileLength);
            float not_value = -1;
            // 获取大于90的数
            float big_scalar = -90;
            AscendC::Adds(yBigSecLocal,yHighLocal,big_scalar,real_tileLength);
            AscendC::Relu(yBigSecLocal,yBigSecLocal,real_tileLength);
            float small_value = 1e-38;
            AscendC::Adds(yBigSignSecLocal,yBigSecLocal,small_value,real_tileLength);
            AscendC::Div(yBigSecLocal,yBigSecLocal,yBigSignSecLocal,real_tileLength);
            AscendC::Mul(yBigSecLocal,yHighLocal,yBigSecLocal,real_tileLength);
            // 获取大于44小于90的数
            float mid_scalar = -45;
            AscendC::Adds(yMidSecLocal,yHighLocal,mid_scalar,real_tileLength);
            AscendC::Relu(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yHighLocal,yMidSignSecLocal,real_tileLength);
            // 获取小于44的数
            AscendC::Sub(ySmallSecLocal,yHighLocal,yMidSecLocal,real_tileLength);
            AscendC::Adds(ySmallSignSecLocal,ySmallSecLocal,small_value,real_tileLength);
            AscendC::Div(ySmallSignSecLocal,ySmallSecLocal,ySmallSignSecLocal,real_tileLength);
            // 获取大于44小于90的数
            AscendC::Sub(yMidSecLocal,yMidSecLocal,yBigSecLocal,real_tileLength);
            AscendC::Adds(yMidSignSecLocal,yMidSecLocal,small_value,real_tileLength);
            AscendC::Div(yMidSignSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            
            // 处理小于44的数
            AscendC::Exp(yTmpLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(yTmpLocal,yTmpLocal,yTmpLocal,real_tileLength);
            float add_scalar = 1;
            AscendC::Adds(yTmpLocal,yTmpLocal,add_scalar,real_tileLength);
            AscendC::Ln(yTmpLocal,yTmpLocal,real_tileLength);
            AscendC::Sub(ySmallSecLocal,ySmallSecLocal,yTmpLocal,real_tileLength);
            AscendC::Exp(ySmallSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(ySmallSecLocal,ySmallSignSecLocal,ySmallSecLocal,real_tileLength);
            // 处理大于45小于90的数
            AscendC::Exp(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Reciprocal(yMidSecLocal,yMidSecLocal,real_tileLength);
            AscendC::Mul(yMidSecLocal,yMidSecLocal,yMidSignSecLocal,real_tileLength);
            // 两者相加
            AscendC::Add(yHighLocal,yMidSecLocal,ySmallSecLocal,real_tileLength);
            AscendC::Mul(yHighLocal,yHighLocal,dyHighLocal,real_tileLength);
            float mul_scalar = 2;
            AscendC::Muls(yHighLocal,yHighLocal,mul_scalar,real_tileLength);
            AscendC::Cast(zLocal,yHighLocal,RoundMode::CAST_NONE,real_tileLength);
            outQueueZ.EnQue<DTYPE_Z>(zLocal);
            inQueueY.FreeTensor(yLocal);
            inQueueDY.FreeTensor(dyLocal);
            inQueueHighTmpY.FreeTensor(yHighLocal);
            inQueueHighTmpY.FreeTensor(dyHighLocal);
            inQueueHighTmpY.FreeTensor(yTmpLocal);
            inQueueHighTmpY.FreeTensor(ySmallSecLocal);
            inQueueHighTmpY.FreeTensor(ySmallSignSecLocal);
            inQueueHighTmpY.FreeTensor(yMidSecLocal);
            inQueueHighTmpY.FreeTensor(yMidSignSecLocal);
            inQueueHighTmpY.FreeTensor(yBigSecLocal);
            inQueueHighTmpY.FreeTensor(yBigSignSecLocal);
        }
    }


private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueDY;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTmpY;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueHighTmpY;
    // TBuf<TPosition::VECCALC>calcBuf;

    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_DY> dyGm;
    GlobalTensor<DTYPE_Z> zGm;

    uint32_t daType;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t lastTileLength;
};




extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAsinhGrad op;
    op.Init(y, dy,z, tiling_data.daType, tiling_data.totalLength,tiling_data.blockLength,tiling_data.tileNum,tiling_data.tileLength,tiling_data.lastTileLength);
    op.Process();
    // TODO: user kernel impl
}