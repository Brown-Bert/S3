#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelAsinh{
public:
    __aicore__ inline KernelAsinh(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t daType, uint32_t totalLength,uint32_t blockLength,uint32_t tileNum,uint32_t tileLength,uint32_t lastTileLength)
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
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), 
        this->blockLength);

        // inQueueTmpX
        if(this->daType == 2)
        {
            pipe.InitBuffer(inQueueHighTmpX, 5, this->tileLength * sizeof(float));
        }
        else if(this->daType == 4)
        {
            pipe.InitBuffer(inQueueTmpX, 4, this->tileLength * sizeof(DTYPE_X));
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], real_tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute1(int32_t progress,uint32_t real_tileLength)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        LocalTensor<DTYPE_X> absLocal = inQueueTmpX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_X> absTmpLocal = inQueueTmpX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_X> signLocal = inQueueTmpX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_X> bigLocal = inQueueTmpX.AllocTensor<DTYPE_X>();

        AscendC::Abs(absLocal,xLocal,real_tileLength);
        AscendC::Div(xLocal,absLocal,xLocal,real_tileLength);

        DTYPE_X big_value = -100;
        AscendC::Adds(signLocal,absLocal,big_value,real_tileLength);
        AscendC::Relu(signLocal,signLocal,real_tileLength);
        DTYPE_X small_value = 10e-10;
        AscendC::Adds(absTmpLocal,signLocal,small_value,real_tileLength);
        AscendC::Div(signLocal,signLocal,absTmpLocal,real_tileLength);

        // 获取大于big_value的数
        AscendC::Mul(bigLocal,absLocal,signLocal,real_tileLength);
        // 获取小于big_value的数
        AscendC::Sub(absLocal,absLocal,bigLocal,real_tileLength);

        // 大于big_value的就执行ln操作
        DTYPE_X scalar = 2;
        AscendC::Muls(bigLocal,bigLocal,scalar,real_tileLength);
        AscendC::Ln(bigLocal,bigLocal,real_tileLength);
        AscendC::Relu(bigLocal,bigLocal,real_tileLength);
        // 小于big_value的就执行下面的操作
        AscendC::Mul(absTmpLocal,absLocal,absLocal,real_tileLength);
        DTYPE_X add_scalar = 1;
        AscendC::Adds(absTmpLocal,absTmpLocal,add_scalar,real_tileLength);
        AscendC::Sqrt(absTmpLocal,absTmpLocal,real_tileLength);
        AscendC::Add(absTmpLocal,absLocal,absTmpLocal,real_tileLength);
        AscendC::Ln(absTmpLocal,absTmpLocal,real_tileLength);               
        // 两者相加
        AscendC::Add(yLocal,absTmpLocal,bigLocal,real_tileLength);
        AscendC::Mul(yLocal,yLocal,xLocal,real_tileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueTmpX.FreeTensor(absLocal);
        inQueueTmpX.FreeTensor(absTmpLocal);
        inQueueTmpX.FreeTensor(signLocal);
        inQueueTmpX.FreeTensor(bigLocal);
        
    }
    __aicore__ inline void Compute(int32_t progress,uint32_t real_tileLength)
    {
        if(this->daType == 2)
        {
            LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
            LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

            // test
            // AscendC::Abs(yLocal,xLocal,real_tileLength);

            LocalTensor<float> xHighLocal = inQueueHighTmpX.AllocTensor<float>();
            LocalTensor<float> absLocal = inQueueHighTmpX.AllocTensor<float>();
            LocalTensor<float> absTmpLocal = inQueueHighTmpX.AllocTensor<float>();
            LocalTensor<float> signLocal = inQueueHighTmpX.AllocTensor<float>();
            LocalTensor<float> bigLocal = inQueueHighTmpX.AllocTensor<float>();
            
            // LocalTensor<half> xLocal2 = xLocal.ReinterpretCast<half>();
            // AscendC::LocalTensor<float> x1Local_temp = x1Local.ReinterpretCast<float>();
            AscendC::Cast(xHighLocal,xLocal,RoundMode::CAST_NONE,real_tileLength);

            AscendC::Abs(absLocal,xHighLocal,real_tileLength);
            AscendC::Div(xHighLocal,absLocal,xHighLocal,real_tileLength);

            float big_value = -100;
            AscendC::Adds(signLocal,absLocal,big_value,real_tileLength);
            AscendC::Relu(signLocal,signLocal,real_tileLength);
            float small_value = 10e-10;
            AscendC::Adds(absTmpLocal,signLocal,small_value,real_tileLength);
            AscendC::Div(signLocal,signLocal,absTmpLocal,real_tileLength);

            // 获取大于big_value的数
            AscendC::Mul(bigLocal,absLocal,signLocal,real_tileLength);
            // 获取小于big_value的数
            AscendC::Sub(absLocal,absLocal,bigLocal,real_tileLength);

            // 大于big_value的就执行ln操作
            float scalar = 2.0;
            AscendC::Muls(bigLocal,bigLocal,scalar,real_tileLength);
            AscendC::Ln(bigLocal,bigLocal,real_tileLength);
            AscendC::Relu(bigLocal,bigLocal,real_tileLength);
            // 小于big_value的就执行下面的操作
            AscendC::Mul(absTmpLocal,absLocal,absLocal,real_tileLength);
            float add_scalar = 1.0;
            AscendC::Adds(absTmpLocal,absTmpLocal,add_scalar,real_tileLength);
            AscendC::Sqrt(absTmpLocal,absTmpLocal,real_tileLength);
            AscendC::Add(absTmpLocal,absLocal,absTmpLocal,real_tileLength);
            AscendC::Ln(absTmpLocal,absTmpLocal,real_tileLength);               
            // 两者相加
            AscendC::Add(absTmpLocal,absTmpLocal,bigLocal,real_tileLength);
            AscendC::Mul(absTmpLocal,absTmpLocal,xHighLocal,real_tileLength);


            AscendC::Cast(yLocal,absTmpLocal,RoundMode::CAST_NONE,real_tileLength);

            outQueueY.EnQue<DTYPE_Y>(yLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueHighTmpX.FreeTensor(xHighLocal);
            inQueueHighTmpX.FreeTensor(absLocal);
            inQueueHighTmpX.FreeTensor(absTmpLocal);
            inQueueHighTmpX.FreeTensor(signLocal);
            inQueueHighTmpX.FreeTensor(bigLocal);
        }
        else if(this->daType == 4)
        {
            LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
            LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            LocalTensor<DTYPE_X> absLocal = inQueueTmpX.AllocTensor<DTYPE_X>();
            LocalTensor<DTYPE_X> absTmpLocal = inQueueTmpX.AllocTensor<DTYPE_X>();
            LocalTensor<DTYPE_X> signLocal = inQueueTmpX.AllocTensor<DTYPE_X>();
            LocalTensor<DTYPE_X> bigLocal = inQueueTmpX.AllocTensor<DTYPE_X>();

            AscendC::Abs(absLocal,xLocal,real_tileLength);
            AscendC::Div(xLocal,absLocal,xLocal,real_tileLength);

            DTYPE_X big_value = -100;
            AscendC::Adds(signLocal,absLocal,big_value,real_tileLength);
            AscendC::Relu(signLocal,signLocal,real_tileLength);
            DTYPE_X small_value = 10e-10;
            AscendC::Adds(absTmpLocal,signLocal,small_value,real_tileLength);
            AscendC::Div(signLocal,signLocal,absTmpLocal,real_tileLength);

            // 获取大于big_value的数
            AscendC::Mul(bigLocal,absLocal,signLocal,real_tileLength);
            // 获取小于big_value的数
            AscendC::Sub(absLocal,absLocal,bigLocal,real_tileLength);

            // 大于big_value的就执行ln操作
            DTYPE_X scalar = 2;
            AscendC::Muls(bigLocal,bigLocal,scalar,real_tileLength);
            AscendC::Ln(bigLocal,bigLocal,real_tileLength);
            AscendC::Relu(bigLocal,bigLocal,real_tileLength);
            // 小于big_value的就执行下面的操作
            AscendC::Mul(absTmpLocal,absLocal,absLocal,real_tileLength);
            DTYPE_X add_scalar = 1;
            AscendC::Adds(absTmpLocal,absTmpLocal,add_scalar,real_tileLength);
            AscendC::Sqrt(absTmpLocal,absTmpLocal,real_tileLength);
            AscendC::Add(absTmpLocal,absLocal,absTmpLocal,real_tileLength);
            AscendC::Ln(absTmpLocal,absTmpLocal,real_tileLength);               
            // 两者相加
            AscendC::Add(yLocal,absTmpLocal,bigLocal,real_tileLength);
            AscendC::Mul(yLocal,yLocal,xLocal,real_tileLength);

            outQueueY.EnQue<DTYPE_Y>(yLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueTmpX.FreeTensor(absLocal);
            inQueueTmpX.FreeTensor(absTmpLocal);
            inQueueTmpX.FreeTensor(signLocal);
            inQueueTmpX.FreeTensor(bigLocal);
        }
    }
    __aicore__ inline void CopyOut(int32_t progress,uint32_t real_tileLength)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, real_tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTmpX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueHighTmpX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;


    // TQue<QuePosition::VECIN, BUFFER_NUM> tmpQueue;
    // TBuf<TPosition::VECCALC>calcBuf;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    uint32_t daType;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t lastTileLength;
};


extern "C" __global__ __aicore__ void asinh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAsinh op;
    op.Init(x, y, tiling_data.daType, tiling_data.totalLength,tiling_data.blockLength,tiling_data.tileNum,tiling_data.tileLength,tiling_data.lastTileLength);
    op.Process();
}