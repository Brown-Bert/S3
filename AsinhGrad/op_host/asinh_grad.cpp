
#include "asinh_grad_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
// 宏定义
const uint32_t BLOCK_SIZE = 32;
const uint32_t ORI_TILE_LENGTH = 32;
const uint32_t MAX_TILE_LENGTH = 1024*4;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    AsinhGradTilingData tiling;
    auto dt = context->GetInputTensor(0)->GetDataType();
    auto dt1 = context->GetInputTensor(1)->GetDataType();
    printf("type: %d %d \n",dt,dt1);
    uint32_t total_length = context->GetInputTensor(0)->GetShapeSize();
    printf("total_hostLengh_ori: %d\n",total_length);
    uint32_t data_type = 0;
    switch(dt)
    {
        case 0:{
            // float
            data_type = 4;
            break;
        }
        case 1:{
            // half
            data_type = 2;
            break;
        }
        case 2:{
            // int 8
            data_type = 1;
            break;
        }
        case 3:{
            // int32
            data_type = 4;
            break;
        }
    }
    uint32_t align_num = BLOCK_SIZE / data_type;
    uint32_t total_length_align = ((total_length + align_num -1) / align_num)*align_num;
    printf("total_hostLengtalign: %d\n",total_length_align);
    
    // 硬件信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();  // vector core num  1
    context->SetBlockDim(aivNum);

    // 计算tiling结构体中所需要的值
    uint32_t block_length = total_length_align / aivNum;
    uint32_t tile_length;
    tile_length = block_length<ORI_TILE_LENGTH?block_length:ORI_TILE_LENGTH;
    // 在满足 block_length / tile_length 接近 8 的条件下进行调整
    while(block_length/tile_length>8)
    {
        tile_length = tile_length *2;
        if(tile_length>=MAX_TILE_LENGTH)
        {
            break;
        }
    }
    uint32_t last_tile_length = block_length % tile_length;
    // 这里改一下
    uint32_t tile_num = block_length / tile_length;
    printf("循环次数%d\n",tile_num);
    // 设置回tiling结构体
    tiling.set_daType(data_type);
    tiling.set_totalLength(total_length_align);
    tiling.set_blockLength(block_length);
    tiling.set_tileNum(tile_num);
    tiling.set_tileLength(tile_length);
    tiling.set_lastTileLength(last_tile_length);

    printf("last_tile_length: %d\n",last_tile_length);


    // 进行保存并更新context
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), 
    context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class AsinhGrad : public OpDef {
public:
    explicit AsinhGrad(const char* name) : OpDef(name)
    {
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(AsinhGrad);
}
