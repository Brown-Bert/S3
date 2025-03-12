
#include "softmax_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    SoftmaxTilingData tiling;
    auto dim = context->GetAttrs()->GetInt(0);
    std::cout<<"dim:"<<*dim<<std::endl;
    auto dim_num = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    std::cout<<"dim_num:"<<dim_num<<std::endl;
    
    auto dt = context->GetInputTensor(0)->GetDataType();
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
    }


    auto batch_size = context->GetInputShape(0)->GetOriginShape().GetDim(0);
    auto height = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    auto width = context->GetInputShape(0)->GetOriginShape().GetDim(dim_num-1);

    // 长度对齐（先不对齐，真对齐了也处理不了）
    uint32_t total_length = context->GetInputTensor(0)->GetShapeSize();
    printf("total_hostLengh_ori: %d\n",total_length);
    uint32_t blockLength = total_length / 1;

    auto forelength = total_length / width;
    // 硬件信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();  // vector core num  1
    context->SetBlockDim(aivNum);
    // 设置
    tiling.set_dim(*dim);
    tiling.set_dim_num(dim_num);
    tiling.set_daType(data_type);
    tiling.set_blockLength(blockLength);
    tiling.set_batch_size(batch_size);
    tiling.set_height(height);
    tiling.set_forelength(forelength);
    tiling.set_width(width);

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
class Softmax : public OpDef {
public:
    explicit Softmax(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Softmax);
}
