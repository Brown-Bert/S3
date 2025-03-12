
#include "non_max_suppression_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>

// 宏定义
const uint32_t BLOCK_SIZE = 32;
const uint32_t ORI_TILE_LENGTH = 32;
const uint32_t MAX_TILE_LENGTH = 1024*4;
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    NonMaxSuppressionTilingData tiling;
    auto center_point_box =  context->GetAttrs()->GetInt(0);
    printf("wai center_point_box: %d\n",*center_point_box);

    auto batch_size = context->GetInputShape(1)->GetOriginShape().GetDim(0);
    auto num_class = context->GetInputShape(1)->GetOriginShape().GetDim(1);
    auto num_box = context->GetInputShape(1)->GetOriginShape().GetDim(2);
    // 还要把num_box对齐
    auto num_box_alin = (num_box*4+31)/32;
    num_box_alin = num_box_alin*32/4;

    uint32_t selected_indices_size = context->GetInputTensor(5)->GetShapeSize();
    selected_indices_size = selected_indices_size/3;
    printf("selected_indices_size: %d\n",selected_indices_size);

    std::cout<<"batch_size:"<<batch_size<<std::endl;
    std::cout<<"num_class:"<<num_class<<std::endl;
    std::cout<<"num_box:"<<num_box<<std::endl;
    std::cout<<"num_box_alin:"<<num_box_alin<<std::endl;
    auto max_per = context->GetInputTensor(2)->GetShapeSize();
    auto iou = context->GetInputTensor(3)->GetShapeSize();
    auto score_sh = context->GetInputTensor(4)->GetShapeSize();
    std::cout<<"max_per:"<<max_per<<std::endl;
    std::cout<<"iou:"<<iou<<std::endl;
    std::cout<<"score_sh:"<<score_sh<<std::endl;
    
    // 硬件信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    auto aivNum = ascendcPlatform.GetCoreNum();  // vector core num  1
    context->SetBlockDim(aivNum);

    // 设置回tiling结构体
    tiling.set_selected_indices_size(selected_indices_size);
    tiling.set_batch_size(batch_size);
    tiling.set_num_class(num_class);
    tiling.set_num_box(num_box);
    tiling.set_num_box_alin(num_box_alin);
    tiling.set_center_point_box(*center_point_box);
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
class NonMaxSuppression : public OpDef {
public:
    explicit NonMaxSuppression(const char* name) : OpDef(name)
    {
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("max_output_boxes_per_class")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("iou_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("score_threshold")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("selected_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("center_point_box").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(NonMaxSuppression);
}
