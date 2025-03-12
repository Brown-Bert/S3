#include "kernel_operator.h"
using namespace AscendC;
#include<math.h>
constexpr int32_t BUFFER_NUM = 1;

class KernelSoftmax{
public:
    __aicore__ inline KernelSoftmax(){}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
     int32_t dim,int32_t dim_num,uint32_t daType,uint32_t blockLength,
     uint32_t batch_size,uint32_t height,uint32_t forelength,uint32_t width)
     {
        this->dim = dim;
        this->dim_num = dim_num;
        this->dataType = daType;
        this->blockLength = blockLength;
        this->batch_size = batch_size;
        this->height = height;
        this->width = width;
        this->forelength = forelength;
        this->height_alin = ((this->height*sizeof(DTYPE_X)+31)/32)*32/sizeof(DTYPE_X);
        this->width_alin = ((this->width*sizeof(DTYPE_X)+31)/32)*32/sizeof(DTYPE_X);
        this->batch_size_alin = ((this->batch_size*sizeof(DTYPE_X)+31)/32)*32/sizeof(DTYPE_X);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(), 
        this->blockLength);
        if(this->dataType == 4)
        {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(inQueueWORK, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(inQueueSUM, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(inQueueSIGN, BUFFER_NUM, this->height_alin * sizeof(float));
            pipe.InitBuffer(inQueueINDEX, BUFFER_NUM, 2 * sizeof(float));
        }
        else if(this->dataType == 2)
        {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->width_alin * sizeof(DTYPE_X));
            pipe.InitBuffer(inQueueHX, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(inQueueWORK, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(inQueueSUM, BUFFER_NUM, this->width_alin * sizeof(float));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->width_alin * sizeof(DTYPE_Y));
            pipe.InitBuffer(inQueueSIGN, BUFFER_NUM, this->height_alin * sizeof(float));
            pipe.InitBuffer(inQueueINDEX, BUFFER_NUM, 2 * sizeof(float));
        }
     }
private:
    TPipe pipe;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    
    // 输入
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueHX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWORK;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSUM;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueINDEX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSIGN;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // 参数设置
    int32_t dim;
    int32_t dim_num;
    uint32_t dataType;
    uint32_t blockLength;
    uint32_t batch_size;
    uint32_t height;
    uint32_t width;
    uint32_t height_alin;
    uint32_t width_alin;
    uint32_t batch_size_alin;
    uint32_t forelength;
private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->width], this->width);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->width], yLocal, this->width);
        outQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void Compute_Seq(uint32_t progress)
    {   
        if(this->dataType == 4)
        {
            AscendC::printf("comdimfu\n");
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> workLocal = inQueueWORK.AllocTensor<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<float> sumLocal = inQueueSUM.AllocTensor<float>();
            LocalTensor<float> indexLocal = inQueueINDEX.AllocTensor<float>();
            // AscendC::ReduceMax<float>(indexLocal,xLocal,workLocal,this->width);
            // float max_value = indexLocal.GetValue(0);
            // max_value = -max_value;
            // AscendC::Adds(xLocal,xLocal,max_value,this->width);
            AscendC::Exp(xLocal,xLocal,this->width);
            AscendC::ReduceSum<float>(sumLocal, xLocal, workLocal,this->width);
            float sum_tmp = sumLocal.GetValue(0);
            AscendC::Duplicate(yLocal,sum_tmp,this->width);
            AscendC::Div(yLocal,xLocal,yLocal,this->width);
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueWORK.FreeTensor(workLocal);
            inQueueSUM.FreeTensor(sumLocal);
            inQueueINDEX.FreeTensor(indexLocal);
        }
        else if(this->dataType == 2)
        {
            // AscendC::printf("this->dataType == 2\n");
            LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
            LocalTensor<float> hxLocal = inQueueHX.AllocTensor<float>();
            LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            LocalTensor<float> workLocal = inQueueWORK.AllocTensor<float>();
            LocalTensor<float> sumLocal = inQueueSUM.AllocTensor<float>();
            LocalTensor<float> indexLocal = inQueueINDEX.AllocTensor<float>();
            AscendC::Cast(hxLocal,xLocal,RoundMode::CAST_NONE,this->width);
            // AscendC::printf("Cast\n");
            // AscendC::ReduceMax<float>(indexLocal,hxLocal,workLocal,this->width);
            // float max_value = indexLocal.GetValue(0);
            // max_value = -max_value;
            // AscendC::Adds(hxLocal,hxLocal,max_value,this->width);
            AscendC::Exp(hxLocal,hxLocal,this->width);
            AscendC::ReduceSum<float>(sumLocal, hxLocal, workLocal,this->width);
            // AscendC::Reciprocal(sumLocal,sumLocal,this->width);
            float sum_tmp = 1/sumLocal.GetValue(0);
            // AscendC::Duplicate(workLocal,sum_tmp,this->width);
            // AscendC::printf("sum_tmp:%f\n",sum_tmp);
            // AscendC::Div(hxLocal,hxLocal,workLocal,this->width);
            AscendC::Muls(hxLocal,hxLocal,sum_tmp,this->width);
            AscendC::Cast(yLocal,hxLocal,RoundMode::CAST_NONE,this->width);

            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueWORK.FreeTensor(workLocal);
            inQueueSUM.FreeTensor(sumLocal);
            inQueueHX.FreeTensor(hxLocal);
            inQueueINDEX.FreeTensor(indexLocal);
        }
    }
    // Dim-1Temp
    __aicore__ inline void CopyInT(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DTYPE_X scalar = 0;
        auto size = xLocal.GetSize();
        AscendC::printf("rellen:%d\n",size);
        AscendC::Duplicate<DTYPE_X>(xLocal, scalar, size);
        for(int i=0;i<this->width;i++)
        {
            DTYPE_X value = xGm.GetValue(progress*this->width+i);
            xLocal.SetValue(i,value);
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyOutT(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        for(int i=0;i<this->width;i++)
        {
            DTYPE_Y tmp_value = yLocal.GetValue(i);
            // AscendC::printf(" yLocal.GetValue(i):%f",tmp_value);
            yGm.SetValue(progress*this->width+i,tmp_value);
        }
        outQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void Compute_NonSeq(uint32_t progress,uint32_t length)
    {
        if(this->dataType == 4)
        {
            LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            LocalTensor<float> workLocal = inQueueWORK.AllocTensor<float>();
            LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            LocalTensor<float> sumLocal = inQueueSUM.AllocTensor<float>();
            LocalTensor<float> signLocal = inQueueSIGN.AllocTensor<float>();
            LocalTensor<float> indexLocal = inQueueINDEX.AllocTensor<float>();
            float scalar = 1e-40;
            AscendC::ReduceMax<float>(indexLocal,xLocal,workLocal,length);
            float max_value = indexLocal.GetValue(0);
            max_value = -max_value;
            AscendC::printf("max_value:%f\n",max_value);
            AscendC::Adds(signLocal,xLocal,scalar,length);
            AscendC::Div(signLocal,xLocal,signLocal,length);
            AscendC::Adds(xLocal,xLocal,max_value,length);
            AscendC::Exp(xLocal,xLocal,length);
            AscendC::Mul(xLocal,xLocal,signLocal,length);
            AscendC::ReduceSum<float>(sumLocal, xLocal, workLocal,length);
            float sum_ori = sumLocal.GetValue(0);
            // AscendC::printf("sum_ori:%f\n",sum_ori);
            AscendC::Duplicate(yLocal,sum_ori,length);
            AscendC::Div(yLocal,xLocal,yLocal,length);
            
            outQueueY.EnQue(yLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueWORK.FreeTensor(workLocal);
            inQueueSUM.FreeTensor(sumLocal);
            inQueueSIGN.FreeTensor(signLocal);
            inQueueINDEX.FreeTensor(indexLocal);
        }
        else if(this->dataType == 2)
        {
            // AscendC::printf("this->dataType == 2\n");
            LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
            LocalTensor<float> hxLocal = inQueueHX.AllocTensor<float>();
            LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
            LocalTensor<float> workLocal = inQueueWORK.AllocTensor<float>();
            LocalTensor<float> sumLocal = inQueueSUM.AllocTensor<float>();
            LocalTensor<float> signLocal = inQueueSIGN.AllocTensor<float>();
            LocalTensor<float> indexLocal = inQueueINDEX.AllocTensor<float>();
            AscendC::Cast(hxLocal,xLocal,RoundMode::CAST_NONE,length);
            // AscendC::printf("Cast\n");
            float scalar = 1e-40;
            AscendC::ReduceMax<float>(indexLocal,hxLocal,workLocal,length);
            float max_value = indexLocal.GetValue(0);
            max_value = -max_value;
            AscendC::printf("max_value:%f\n",max_value);
            AscendC::Adds(signLocal,hxLocal,scalar,length);
            AscendC::Div(signLocal,hxLocal,signLocal,length);
            AscendC::Adds(hxLocal,hxLocal,max_value,length);
            AscendC::Exp(hxLocal,hxLocal,length);
            AscendC::Mul(hxLocal,hxLocal,signLocal,length);
            AscendC::ReduceSum<float>(sumLocal, hxLocal, workLocal,length);
            float sum_ori = sumLocal.GetValue(0);
            
            AscendC::Duplicate(workLocal,sum_ori,length);
            AscendC::Div(workLocal,hxLocal,workLocal,length);
            AscendC::Cast(yLocal,workLocal,RoundMode::CAST_NONE,length);

            outQueueY.EnQue(yLocal);
            inQueueHX.FreeTensor(hxLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueWORK.FreeTensor(workLocal);
            inQueueSUM.FreeTensor(sumLocal);
            inQueueSIGN.FreeTensor(signLocal);
            inQueueINDEX.FreeTensor(indexLocal);
        }
        return ;
    }
public:
    __aicore__ inline void Process()
    {
        AscendC::printf("this->width*sizeof(DTYPE_X):%d\n",this->width*sizeof(DTYPE_X));
        // if((this->width*sizeof(DTYPE_X))%32==0)
        if(this->dim == -1 || (this->dim == this->dim_num-1))
        {
            AscendC::printf("this->width*sizeof(DTYPE_X))%32==0\n");
            if((this->width*sizeof(DTYPE_X))%32==0)
            {
                for(uint32_t i=0;i<this->forelength;i++)
                {
                    CopyIn(i);
                    Compute_Seq(i);
                    CopyOut(i);
                }
            }
            else{
                AscendC::printf("this->width*sizeof(DTYPE_X))%32!=0\n");
                for(uint32_t i=0;i<this->forelength;i++)
                {
                    CopyInT(i);
                    Compute_NonSeq(i,this->width_alin);
                    CopyOutT(i);
                }
            }
        }
        }
};

extern "C" __global__ __aicore__ void softmax(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x,y,tiling_data.dim,tiling_data.dim_num,tiling_data.daType,
    tiling_data.blockLength,tiling_data.batch_size,tiling_data.height,tiling_data.forelength,tiling_data.width);
    op.Process();
    // TODO: user kernel impl
}