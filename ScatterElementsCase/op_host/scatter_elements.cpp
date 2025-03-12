
#include "scatter_elements_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  ScatterElementsTilingData tiling;
  auto axis =  context->GetAttrs()->GetInt(0);
  const char *reduce = context->GetAttrs()->GetAttrPointer<char>(1);
  // 0：覆盖，1：add，2：multiply
  uint32_t optype = 0;
  if(strcmp(reduce,"add")==0)
  {
      optype = 1;
  }
  else if(strcmp(reduce, "multiply") == 0)
  {
      optype = 2;
  }

  auto shape = context->GetInputShape(0)->GetStorageShape();
  auto dim_num = shape.GetDimNum();

  // 根据axis要更新的维度，计算出要搬运的数据之间的间隔
  uint32_t interval = 0;

  // 获取数据类型
  auto dt = context->GetInputTensor(0)->GetDataType();

  // 获取UB内存大小
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  // 获取AiCore的物理核数
  auto coreNum = ascendcPlatform.GetCoreNum();

  // 在compute接口中，每次计算需要同时消耗多少个LocalTensor
  uint32_t ubDataNum = 8; // 是根据具体的代码逻辑设定的，不是通过算出来的

  // 通过context获取用户传入的数据大小
  uint32_t inputNum = 0;
  if (dt == 1){
    inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  }else if (dt == 0 && dim_num == 2){
    inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    auto dim0 = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    auto dim1 = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    interval = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    tiling.set_dim0(dim0);
    tiling.set_dim1(dim1);
  }else if (dt == 0 && dim_num == 3){
    inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    auto dim0 = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    auto dim1 = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    auto dim2 = context->GetInputShape(1)->GetStorageShape().GetDim(2);
    auto varDim0 = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto varDim1 = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    auto varDim2 = context->GetInputShape(0)->GetStorageShape().GetDim(2);
    interval = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2);
    tiling.set_dim0(dim0);
    tiling.set_dim1(dim1);
    tiling.set_dim2(dim2);
    tiling.set_varDim0(varDim0);
    tiling.set_varDim1(varDim1);
    tiling.set_varDim2(varDim2);
  }else if(dt == 3){
    inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    auto dim0 = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    auto dim1 = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    auto dim2 = context->GetInputShape(1)->GetStorageShape().GetDim(2);
    auto varDim0 = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto varDim1 = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    auto varDim2 = context->GetInputShape(0)->GetStorageShape().GetDim(2);
    tiling.set_dim0(dim0);
    tiling.set_dim1(dim1);
    tiling.set_dim2(dim2);
    tiling.set_varDim0(varDim0);
    tiling.set_varDim1(varDim1);
    tiling.set_varDim2(varDim2);
  }else if (dt == 4){
    inputNum = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
    auto dim0 = context->GetInputShape(1)->GetStorageShape().GetDim(0);
    auto dim1 = context->GetInputShape(1)->GetStorageShape().GetDim(1);
    auto dim2 = context->GetInputShape(1)->GetStorageShape().GetDim(2);
    auto dim3 = context->GetInputShape(1)->GetStorageShape().GetDim(3);
    auto varDim0 = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto varDim1 = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    auto varDim2 = context->GetInputShape(0)->GetStorageShape().GetDim(2);
    auto varDim3 = context->GetInputShape(0)->GetStorageShape().GetDim(3);
    tiling.set_dim0(dim0);
    tiling.set_dim1(dim1);
    tiling.set_dim2(dim2);
    tiling.set_dim3(dim3);
    tiling.set_varDim0(varDim0);
    tiling.set_varDim1(varDim1);
    tiling.set_varDim2(varDim2);
    tiling.set_varDim3(varDim3);
  }else{
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  }

  // 用户输入数据的类型大小（以字节为单位）
  uint32_t inputBytes = 0;
  ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputBytes);

  // 数据的总大小
  uint32_t inputLength = inputNum * inputBytes;


  // aicore每次最多处理多少block块
  uint32_t blockNum = ubSize / 32 / 2 / ubDataNum;

  // 根据每次最多处理的block块，转换成每次最多处理的数据个数
  uint32_t dataNum = blockNum * 32 / inputBytes;

  // 根据32字节对齐计算数据的字节数
  uint32_t inputLengthAlgin32 = (inputLength + 32 - 1) / 32 * 32;

  tiling.set_ridOfNum((inputLengthAlgin32 - inputLength) / inputBytes);

  // 计算数据需要几个core去执行，如果数据量太小就不需要全部的core去执行
  coreNum = (coreNum < inputLengthAlgin32 / 32) ? coreNum : inputLengthAlgin32 / 32;
  coreNum = (coreNum >= 1) ? coreNum : 1;

  // 计算每个core需要处理多少个block
  uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / 32 / coreNum;

  // 上面一行不是整除的话，会剩余几个block
  uint32_t tailBlockNum = inputLengthAlgin32 / 32 % coreNum;

  /*
      小核
  */

  // 计算小核要处理的数据量
  uint32_t smallCoreDataNum = everyCoreInputBlockNum * 32 / inputBytes;

  // 计算小核，根据每个小核要处理多少block，但是一次只能处理多少个block，计算出需要计算多少次
  uint32_t smallCoreCount = everyCoreInputBlockNum / blockNum;
  uint32_t smallCoreCarryNum = (everyCoreInputBlockNum % blockNum == 0) ? smallCoreCount : (smallCoreCount + 1);

  // 计算小核最后一次需要处理多少数据
  uint32_t smallCoreFinallDealNum = smallCoreDataNum - (dataNum * smallCoreCount);
  smallCoreFinallDealNum = (smallCoreFinallDealNum == 0) ? dataNum : smallCoreFinallDealNum;

  /**
      大核
   */
  everyCoreInputBlockNum++;
  uint32_t bigCoreDataNum = everyCoreInputBlockNum * 32 / inputBytes;

  uint32_t bigCoreCount = everyCoreInputBlockNum / blockNum;
  uint32_t bigCoreCarryNum = (everyCoreInputBlockNum % blockNum == 0) ? bigCoreCount : (bigCoreCount + 1);

  uint32_t bigCoreFinallDealNum = bigCoreDataNum - (dataNum * bigCoreCount);
  bigCoreFinallDealNum = (bigCoreFinallDealNum == 0) ? dataNum : bigCoreFinallDealNum;

  // 将上述计算的值全部回填到tiling中
  tiling.set_smallCoreDataNum(smallCoreDataNum);
  tiling.set_bigCoreDataNum(bigCoreDataNum);
  tiling.set_smallCoreCarryNum(smallCoreCarryNum);
  tiling.set_bigCoreCarryNum(bigCoreCarryNum);
  tiling.set_tileDataNum(dataNum);
  tiling.set_smallCoreFinallDealNum(smallCoreFinallDealNum);
  tiling.set_bigCoreFinallDealNum(bigCoreFinallDealNum);
  tiling.set_bigCoreNum(tailBlockNum);
  tiling.set_dataType(dt);
  tiling.set_reduce(optype);
  tiling.set_axis(*axis);
  tiling.set_dims(dim_num);
  tiling.set_interval(interval);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  context->SetBlockDim(coreNum);

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    // gert::Shape* y_shape = context->GetOutputShape(0);
    // *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class ScatterElements : public OpDef {
public:
    explicit ScatterElements(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").Int();
        this->Attr("reduce").AttrType(OPTIONAL).String("None");

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(ScatterElements);
}
