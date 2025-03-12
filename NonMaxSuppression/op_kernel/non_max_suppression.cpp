#include "kernel_operator.h"
using namespace AscendC;
#include<math.h>
constexpr int32_t BUFFER_NUM = 1;

class KernelNonMaxSuppression{
public:
     __aicore__ inline KernelNonMaxSuppression(){}
     __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR scores,GM_ADDR max_output_boxes_per_class,
     GM_ADDR iou_threshold,GM_ADDR score_threshold,GM_ADDR selected_indices,
     uint32_t selected_indices_size,
     uint32_t batch_size,uint32_t num_class,uint32_t num_box,uint32_t num_box_alin,
     uint32_t center_point_box,uint32_t loc,int32_t fore_succes_num)
    {   
        this->selected_indices_size = selected_indices_size;
        this->batch_size = batch_size;
        this->num_class = num_class;
        this->num_box = num_box;
        this->num_box_alin = num_box_alin;
        this->loc = loc;

        mpcGm.SetGlobalBuffer((__gm__ int32_t *)max_output_boxes_per_class, sizeof(int32_t));
        this->max_output_boxes_per_class = mpcGm.GetValue(0);
        
        iouGm.SetGlobalBuffer((__gm__ float *)iou_threshold, sizeof(float));
        this->iou_threshold = iouGm.GetValue(0);
        AscendC::printf("iou_threshold:%f\n",this->iou_threshold);
        
        score_thresholdGm.SetGlobalBuffer((__gm__ float *)score_threshold, sizeof(float));
        this->score_threshold = score_thresholdGm.GetValue(0);
        AscendC::printf("max_output_boxes_per_class:%d\n",this->max_output_boxes_per_class);
        // 初始化
        this->center_point_box = center_point_box;
        AscendC::printf("center_point_box:%d\n",center_point_box);
         
        // 初始化输出的大小[这里有问题]
        // auto tmp = this->selected_indices_size/this->batch_size/this->num_class;
        // AscendC::printf("max_output_boxes_per_class_gai:%d\n",tmp);
        // if(tmp!=this->max_output_boxes_per_class)
        // {
        //     this->max_output_boxes_per_class = tmp;
        // }
        // 初始化管道，这里用了doubleBUffer,这里还没改[改了]
        auto battch_num = loc / this->num_class;
        boxesGm.SetGlobalBuffer((__gm__ float *)boxes + battch_num * this->num_box * 4, 
        4*this->num_box_alin);
        scoresGm.SetGlobalBuffer((__gm__ float *)scores + loc * this->num_box, 
        this->num_box_alin);
        selected_indicesGm.SetGlobalBuffer((__gm__ int32_t *)selected_indices+3*fore_succes_num, 
        3*this->max_output_boxes_per_class);
        
        // 这里还没改
        pipe.InitBuffer(inQueueBOXES, BUFFER_NUM, this->num_box_alin * 4 * sizeof(float));
        pipe.InitBuffer(inQueueSCORES, BUFFER_NUM, this->num_box_alin * sizeof(float));
        // 存放索引的
        pipe.InitBuffer(inQueueTEMPINDEX, BUFFER_NUM, 2 * sizeof(float));
        // 临时空间使用
        pipe.InitBuffer(inQueueWORK, 1, this->num_box_alin*sizeof(float));
        // inQueueCLASS
        pipe.InitBuffer(inQueueINDIC, 1, this->max_output_boxes_per_class*sizeof(int32_t)*3);
    }
    __aicore__ inline void Process(int32_t &fore_succes_num)
    {
        CopyIn();
        Compute1();
        CopyOut(fore_succes_num);
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> boxesLocal = inQueueBOXES.AllocTensor<float>();
        AscendC::LocalTensor<float> scoresLocal = inQueueSCORES.AllocTensor<float>();
        AscendC::DataCopy(boxesLocal, boxesGm, 4*this->num_box_alin);
        AscendC::DataCopy(scoresLocal, scoresGm, this->num_box_alin);
        inQueueBOXES.EnQue(boxesLocal);
        inQueueSCORES.EnQue(scoresLocal);
    }
    __aicore__ inline void CopyOut(int32_t &fore_succes_num)
    {
        AscendC::LocalTensor<int32_t> indicLocal = inQueueINDIC.DeQue<int32_t>();
        for(uint32_t i = 0;i<this->sucess_num;i++)
        {
            // if(this->loc*this->max_output_boxes_per_class+i>=this->selected_indices_size)
            // {
            //     return ;
            // }
            auto batch_index = indicLocal.GetValue(i*3+0);
            auto class_index = indicLocal.GetValue(i*3+1);
            auto box_index = indicLocal.GetValue(i*3+2);
            selected_indicesGm.SetValue(i*3+0,batch_index);
            selected_indicesGm.SetValue(i*3+1,class_index);
            selected_indicesGm.SetValue(i*3+2,box_index);
        }
        // AscendC::DataCopy(selected_indicesGm, indicLocal, 3*this->max_output_boxes_per_class);
        inQueueINDIC.FreeTensor(indicLocal);
        fore_succes_num = fore_succes_num+this->sucess_num;
    }
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<int32_t> indicLocal = inQueueINDIC.AllocTensor<int32_t>();
        // int32_t sucess_num = 0;
        AscendC::LocalTensor<float> boxesLocal = inQueueBOXES.DeQue<float>();
        AscendC::LocalTensor<float> scoresLocal = inQueueSCORES.DeQue<float>();
        AscendC::LocalTensor<float> indexLocal = inQueueTEMPINDEX.AllocTensor<float>();
        AscendC::LocalTensor<float> workLocal = inQueueWORK.AllocTensor<float>();
        for(int32_t i = 0;i<this->num_box;i++)
        {
            // 去置信分数最大值
            AscendC::ReduceMax(indexLocal,scoresLocal,workLocal,this->num_box,true);
            float best_score = indexLocal.GetValue(0);
            AscendC::LocalTensor<int32_t> x1Local_temp = indexLocal.ReinterpretCast<int32_t>();
            int32_t index = x1Local_temp.GetValue(1);
            if(best_score<this->score_threshold)
            {
                break;
            }
            // 进行设置
            int32_t batch_index = this->loc/this->num_class;
            int32_t class_index = this->loc % this->num_class;
            AscendC::printf("batch_index class_index index:%d %d %d\n",batch_index,class_index,index);
            // 这里应该要把批次，类别，框索引（idex）设置到一个Tensor中并放入输出tensor中
            indicLocal.SetValue(0+sucess_num*3,batch_index);
            indicLocal.SetValue(1+sucess_num*3,class_index);
            indicLocal.SetValue(2+sucess_num*3,index);
            scoresLocal.SetValue(index,this->score_threshold-1);
            this->sucess_num ++;
            // 进行判断数量是否足够 
            if(this->sucess_num >= this->max_output_boxes_per_class)
            {
                break;
            }
            // 计算iou
            for(int32_t j = 0;j<this->num_box;j++)
            {
                // 获取当前最大索引的坐标值
                float mx1,mx2,my1,my2;
                if(this->center_point_box == 0)
                {
                    mx1 = boxesLocal.GetValue(index*4+0);
                    my1 = boxesLocal.GetValue(index*4+1);
                    mx2 = boxesLocal.GetValue(index*4+2);
                    my2 = boxesLocal.GetValue(index*4+3);
                }
                else if(this->center_point_box == 1)
                {
                    float cx = boxesLocal.GetValue(index*4+0);
                    float cy = boxesLocal.GetValue(index*4+1);
                    float w = boxesLocal.GetValue(index*4+2);
                    float h = boxesLocal.GetValue(index*4+3);
                    mx1 = cx -w/2;
                    my1 = cy - h/2;
                    mx2 = cx + w/2;
                    my2 = cy + h/2;
                }
                // tmpClassLocal
                float tmo_score = scoresLocal.GetValue(j);
                if(tmo_score<this->score_threshold)
                {
                    continue;
                }
                if(this->center_point_box == 0)
                {
                    float x1 = boxesLocal.GetValue(j*4+0);
                    float y1 = boxesLocal.GetValue(j*4+1);
                    float x2 = boxesLocal.GetValue(j*4+2);
                    float y2 = boxesLocal.GetValue(j*4+3);
                    float tmp_iou;
                    compute_iou(x1,y1,x2,y2,mx1,my1,mx2,my2,tmp_iou);
                    // AscendC::printf("compute iou:%f\n",tmp_iou);
                    if(tmp_iou>=this->iou_threshold)
                    {
                        AscendC::printf("tmp_iou>=this->iou_threshold:%d %f %f\n",j,tmp_iou,this->iou_threshold);
                        scoresLocal.SetValue(j,this->score_threshold-1);
                    }
                }
                else if(this->center_point_box == 1)
                {
                    float cx1 = boxesLocal.GetValue(j*4+0);
                    float cy1 = boxesLocal.GetValue(j*4+1);
                    float w1 = boxesLocal.GetValue(j*4+2);
                    float h1 = boxesLocal.GetValue(j*4+3);
                    float x1 = cx1 - w1/2;
                    float y1 = cy1 - h1/2;
                    float x2 = cx1 + w1/2;
                    float y2 = cy1 + h1/2;
                    float tmp_iou;
                    compute_iou(x1,y1,x2,y2,mx1,my1,mx2,my2,tmp_iou);
                    // AscendC::printf("w1 h1:%f %f\n",w1,h1);
                    if(tmp_iou>=this->iou_threshold)
                    {
                        AscendC::printf("tmp_iou>=this->iou_threshold:%d %f %f\n",j,tmp_iou,this->iou_threshold);
                        scoresLocal.SetValue(j,this->score_threshold-1);
                    }
                }
            }
            }
        inQueueBOXES.FreeTensor(boxesLocal);
        inQueueSCORES.FreeTensor(scoresLocal);
        inQueueTEMPINDEX.FreeTensor(indexLocal);
        inQueueWORK.FreeTensor(workLocal);
        inQueueINDIC.EnQue(indicLocal);
    }
    __aicore__ inline void compute_iou(float xa_1,float ya_1,float xa_2,float ya_2,
                float xb_1,float yb_1,float xb_2,float yb_2,float& iou)
                {   
                    // AscendC::printf("zuobiao1:%f,%f,%f,%f\n",xa_1,ya_1,xa_2,ya_2);
                    // AscendC::printf("zuobiao2:%f,%f,%f,%f\n",xb_1,yb_1,xb_2,yb_2);
                    // 他这个坐标的存放不是规则的，需要自己定义左上角和右下角
                    float l_x1 = xa_1 < xa_2?xa_1:xa_2;
                    float t_x1 = ya_1 < ya_2?ya_1:ya_2;
                    float r_x1 = xa_1 > xa_2?xa_1:xa_2;
                    float b_x1 = ya_1 > ya_2?ya_1:ya_2;

                    float l_x2 = xb_1 < xb_2?xb_1:xb_2;
                    float t_x2 = yb_1 < yb_2?yb_1:yb_2;
                    float r_x2 = xb_1 > xb_2?xb_1:xb_2;
                    float b_x2 = yb_1 > yb_2?yb_1:yb_2;

                    float xLeft = l_x1>l_x2?l_x1:l_x2;
                    float yTop = t_x1>t_x2?t_x1:t_x2;
                    float xRight = r_x1<r_x2?r_x1:r_x2;
                    float yBottom = b_x1<b_x2?b_x1:b_x2;
                    float Width = (xRight-xLeft)>0?(xRight-xLeft):0;
                    float Height = (yBottom-yTop)>0?(yBottom-yTop):0;
                    float InterArea = Width*Height;
                    // AscendC::printf("Width,Height:%f %f\n",Width,Height);
                    float A_Area = (b_x1 - t_x1)*(r_x1-l_x1);
                    float B_Area = (b_x2 - t_x2)*(r_x2-l_x2);
                    float UnionArea = A_Area + B_Area - InterArea;
                    iou =  InterArea / UnionArea;
                }
private:
        TPipe pipe;
        GlobalTensor<int32_t> mpcGm;
        GlobalTensor<float> iouGm;
        GlobalTensor<float> score_thresholdGm;
        // 输入
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBOXES;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSCORES;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTEMPINDEX;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWORK;
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueINDIC;
        // 输出
        TQue<QuePosition::VECOUT, BUFFER_NUM> SELECTED_INDICES_Queue;

        GlobalTensor<float> boxesGm;
        GlobalTensor<float> scoresGm;
        GlobalTensor<int32_t> selected_indicesGm;

        // 
        GlobalTensor<int32_t> max_output_boxes_per_class_Gm;
        GlobalTensor<float> iou_threshold_Gm;
        GlobalTensor<float> score_threshold_Gm;

        uint32_t selected_indices_size;
        uint32_t center_point_box;
        // 尺度信息
        uint32_t batch_size;
        uint32_t num_class;
        uint32_t num_box;
        uint32_t num_box_alin;
        // 直接接受globalTensor里面的
        int32_t max_output_boxes_per_class;
        float iou_threshold;
        float score_threshold;

        // 用于定位输出变量拷贝的位置
        uint32_t loc = 0;
        int32_t sucess_num = 0;
};
extern "C" __global__ __aicore__ void non_max_suppression(GM_ADDR boxes, GM_ADDR scores, GM_ADDR max_output_boxes_per_class, 
GM_ADDR iou_threshold, GM_ADDR score_threshold, GM_ADDR selected_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    auto loop_time = tiling_data.batch_size*tiling_data.num_class;
    int32_t fore_succes_num = 0;
    // auto tmp = max_output_boxes_per_class*tiling_data.batch_size*tiling_data.num_class;
    for(uint32_t i =0;i<loop_time;i++)
    {
        KernelNonMaxSuppression op;
        op.Init(boxes, scores,max_output_boxes_per_class,iou_threshold, score_threshold,selected_indices,
        tiling_data.selected_indices_size,
        tiling_data.batch_size,tiling_data.num_class,tiling_data.num_box,tiling_data.num_box_alin,
        tiling_data.center_point_box,i,fore_succes_num
        );
        op.Process(fore_succes_num);
        AscendC::printf("fore_succes_num:%d\n",fore_succes_num);
    }
}