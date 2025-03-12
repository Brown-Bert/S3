
#include "replication_pad2d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ReplicationPad2dTilingData tiling;
    auto dim_num = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    std::cout<<"dim_num:"<<dim_num<<std::endl;
    uint32_t x_total = context->GetInputTensor(0)->GetShapeSize();
    uint32_t y_total = context->GetInputTensor(2)->GetShapeSize();
    std::cout<<"x_total"<<x_total<<std::endl;
    std::cout<<"y_total"<<y_total<<std::endl;
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
    auto param_h = context->GetInputShape(0)->GetOriginShape().GetDim(dim_num-2);
    auto param_w = context->GetInputShape(0)->GetOriginShape().GetDim(dim_num-1);
    auto param_c = x_total/(param_h*param_w);
    // std::cout<<"param_c"<<param_c<<std::endl;
    // std::cout<<"param_h"<<param_h<<std::endl;
    // std::cout<<"param_w"<<param_w<<std::endl;
    // 硬件信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();  // vector core num  1
    context->SetBlockDim(aivNum);

    // 设置
    tiling.set_x_total(x_total);
    tiling.set_y_total(y_total);
    tiling.set_dim_num(dim_num);
    tiling.set_datatype(data_type);
    tiling.set_param_c(param_c);
    tiling.set_param_h(param_h);
    tiling.set_param_w(param_w);

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
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ReplicationPad2d : public OpDef {
public:
    explicit ReplicationPad2d(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
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

OP_ADD(ReplicationPad2d);
}
