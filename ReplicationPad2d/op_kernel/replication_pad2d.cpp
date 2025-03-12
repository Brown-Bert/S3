#include "kernel_operator.h"
using namespace AscendC;
#include<math.h>
constexpr int32_t BUFFER_NUM = 1;
class KernelReplicationPad2d{
public:
     __aicore__ inline KernelReplicationPad2d(){}
     __aicore__ inline void Init(GM_ADDR x, GM_ADDR paddings,GM_ADDR y,
     uint32_t x_total,uint32_t y_total,
     uint32_t dim_num,uint32_t datatype,
     uint32_t param_c,uint32_t param_h,uint32_t param_w)
     {
        this->dim_num = dim_num;
        this->datatype = datatype;
        this->param_c = param_c;
        this->param_h = param_h;
        this->param_w = param_w;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x,sizeof(DTYPE_X)*x_total);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y,sizeof(DTYPE_Y)*y_total);
        pGm.SetGlobalBuffer((__gm__ int32_t *)paddings,sizeof(int32_t)*4);
        // 计算获取对齐数
        this->pad_left = pGm.GetValue(0);
        this->pad_right = pGm.GetValue(1);
        this->pad_top = pGm.GetValue(2);
        this->pad_bottom = pGm.GetValue(3);
        // AscendC::printf("this->pad_left:%d\n",this->pad_left);
        // AscendC::printf("this->pad_right:%d\n",this->pad_right);
        // AscendC::printf("this->pad_top:%d\n",this->pad_top);
        // AscendC::printf("this->pad_bottom:%d\n",this->pad_bottom);
        
        this->w_alin = this->param_w + this->pad_left + this->pad_right;
        this->h_alin = this->param_h + this->pad_top + this->pad_bottom;
        // AscendC::printf("this->w_alin:%d\n",this->w_alin);
        // AscendC::printf("this->h_alin:%d\n",this->h_alin);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->param_w*sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->w_alin*sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueY2, 1, this->w_alin*sizeof(DTYPE_Y));

     }
     __aicore__ inline void Process()
     {
        for(int32_t i=0;i<this->param_c;i++)
        {
            for(int32_t j = 0;j<this->param_h;j++)
            {
                CopyIn(i,j);
                Compute1(i,j);
                CopyOut(i,j);
            }
            for(int32_t k=0;k<this->pad_top+this->pad_bottom;k++)
            {
                Compute2(i,k);
                CopyOut2(i,k);
            }
        }
     }
private:
    TPipe pipe;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<int32_t> pGm;
    GlobalTensor<DTYPE_Y> yGm;

    // 输入
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECIN, BUFFER_NUM> outQueueY2;
    // 参数设置
    uint32_t dim_num;
    uint32_t datatype;
    uint32_t param_c;
    uint32_t param_h;
    uint32_t param_w;
    int32_t pad_left;
    int32_t pad_right;
    int32_t pad_top;
    int32_t pad_bottom;
    int32_t w_alin;
    int32_t h_alin;
private:
    __aicore__ inline void CopyIn(int32_t pc,int32_t loc)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        if((this->param_w*sizeof(DTYPE_X))%32==0)
        {
            AscendC::DataCopy(xLocal,xGm[(pc*this->param_h+loc)*this->param_w],this->param_w);
        }
        else{
            for(int i=0;i<this->param_w;i++)
            {
                auto tmp = xGm.GetValue((pc*this->param_h+loc)*this->param_w+i);
                xLocal.SetValue(i,tmp);
            }
        }
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t pc,int32_t loc)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        for(int32_t i =0;i<this->w_alin;i++)
        {
            auto tmp = yLocal.GetValue(i);
            yGm.SetValue(((pc*this->h_alin+loc+this->pad_top)*this->w_alin+i),tmp);
        }
        // AscendC::DataCopy(yGm[((pc*this->h_alin+loc)*this->w_alin)],yLocal,this->w_alin);
        outQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void Compute1(int32_t pc,int32_t loc)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        for(int32_t i=0;i<this->param_w+this->pad_left+this->pad_right;i++)
        {
            if(i<this->pad_left)
            {
                DTYPE_X tmp = xLocal.GetValue(0);
                yLocal.SetValue(i,tmp);
                continue;
            }
            if(i>=this->param_w+this->pad_left)
            {
                DTYPE_X tmp = xLocal.GetValue(this->param_w-1);
                yLocal.SetValue(i,tmp);
                continue;
            }
            DTYPE_X tmp = xLocal.GetValue(i-this->pad_left);
            yLocal.SetValue(i,tmp);
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline  void Compute2(int32_t p_c,int32_t loc)
    {   
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY2.AllocTensor<DTYPE_Y>();
        if(loc<this->pad_top)
            {
                for(int32_t i =0;i<this->w_alin;i++)
                {
                    auto tmp = yGm.GetValue((p_c*this->h_alin+this->pad_top)*this->w_alin+i);
                    yLocal.SetValue(i,tmp);
                }
            }
            else{
                for(int32_t i =0;i<this->w_alin;i++)
                {
                    auto tmp = yGm.GetValue((p_c*this->h_alin+this->pad_top+this->param_h-1)*this->w_alin+i);
                    // AscendC::printf("loc>this->pad_top tmp:%d %f\n",p_c,tmp);
                    yLocal.SetValue(i,tmp);
                }
                for(int i=0;i<this->w_alin;i++)
                {
                    auto ww = yLocal.GetValue(i);
                    // AscendC::printf("yLocal %f\n",ww);
                }
            }
        outQueueY2.EnQue(yLocal);
    }
    __aicore__ inline  void CopyOut2(int32_t p_c,int32_t loc)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY2.DeQue<DTYPE_Y>();
        if(loc<this->pad_top)
        {
            for(uint32_t i=0;i<this->w_alin;i++)
            {
                auto tmp = yLocal.GetValue(i);
                yGm.SetValue((p_c*this->h_alin+loc)*this->w_alin+i,tmp);
            }
            // AscendC::DataCopy(xGm[p_c*this->w_alin+loc*this->w_alin],yLocal,this->w_alin);
        }
        else{
            for(int32_t i=0;i<this->w_alin;i++)
            {
                // AscendC::printf("CopyOut2 pc,loc:%d %d",p_c,loc);
                auto tmp = yLocal.GetValue(i);
                yGm.SetValue((p_c*this->h_alin+loc+this->param_h)*this->w_alin+i,tmp);
            }
            for(int i=0;i<this->w_alin;i++)
            {
                auto ww = yLocal.GetValue(i);
                // AscendC::printf("yLocal %f\n",ww);
            }
            // AscendC::DataCopy(xGm[p_c*this->w_alin+(loc+this->param_h)*this->w_alin],yLocal,this->w_alin);
        }
        outQueueY2.FreeTensor(yLocal);
    }
};




extern "C" __global__ __aicore__ void replication_pad2d(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelReplicationPad2d op;
    op.Init(x,paddings,y,
    tiling_data.x_total,tiling_data.y_total,
    tiling_data.dim_num,tiling_data.datatype,
    tiling_data.param_c,tiling_data.param_h,tiling_data.param_w);
    op.Process();
}