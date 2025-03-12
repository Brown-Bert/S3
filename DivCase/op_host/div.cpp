
#include "div_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  DivTilingData tiling;
  auto dt = context->GetInputTensor(0)->GetDataType();

  // 计算是否要进行广播以及广播的进行方式
  auto inputShape0 = context->GetInputShape(0)->GetStorageShape();
  auto inputShape1 = context->GetInputShape(1)->GetStorageShape();
  if (inputShape0.GetShapeSize() == inputShape1.GetShapeSize()){
      // 两个输入的数据大小相同，不需要广播
      tiling.set_isBroadcast(0);
  }else {
      // 先记录两个输入的数据的维度详细信息
      // 1、先判断两个输入的维度数量是不是一样的
      auto dim0 = inputShape0.GetDimNum();
      auto dim1 = inputShape1.GetDimNum();
      uint32_t shape00 = 0;
      uint32_t shape01 = 0;
      uint32_t shape10 = 0;
      uint32_t shape11 = 0;
      if (dim0 < dim1){
          shape00 = 1;
          shape01 = inputShape0.GetDim(0);
          shape10 = inputShape1.GetDim(0);
          shape11 = inputShape1.GetDim(1);
      }else if (dim0 > dim1){
          shape00 = inputShape0.GetDim(0);
          shape01 = inputShape0.GetDim(1);
          shape10 = 1;
          shape11 = inputShape1.GetDim(0);
      }else{
          shape00 = inputShape0.GetDim(0);
          shape01 = inputShape0.GetDim(1) * inputShape0.GetDim(2);
          shape10 = inputShape1.GetDim(0);
          shape11 = inputShape1.GetDim(1) * inputShape1.GetDim(2);
      }
      tiling.set_inputShape00(shape00);
      tiling.set_inputShape01(shape01);
      tiling.set_inputShape10(shape10);
      tiling.set_inputShape11(shape11);
    //   std::cout << "shape00 = " << shape00 << " shape01 = " << shape01 << " shape10 = " << shape10 << " shape11 = " << shape11 << std::endl;
      // tiling.set_inputShape00(inputShape0.GetDim(0));
      // tiling.set_inputShape01(inputShape0.GetDim(1));
      // tiling.set_inputShape10(inputShape1.GetDim(0));
      // tiling.set_inputShape11(inputShape1.GetDim(1));
      // 先判断谁需要进行广播
      uint8_t who = -1;
      if (inputShape0.GetShapeSize() < inputShape1.GetShapeSize()){
          // 第一个输入的数据大小小于第二个输入的数据大小，第一个输入需要进行广播
          who = 0;
          tiling.set_who(who);
          // 判断广播的方向
          uint8_t axis = -1;
          if (shape00 == 1){
              axis = 1;
          }else{
              axis = 0;
          }
          tiling.set_axis(axis);
      }else{
          // 第二个输入的数据大小小于第一个输入的数据大小，第二个输入需要进行广播
          who = 1;
          tiling.set_who(who);
          // 判断广播的方向
          uint8_t axis = -1;
          if (shape10 == 1){
              axis = 1;
          }else{
              axis = 0;
          }
          tiling.set_axis(axis);
      }
      tiling.set_isBroadcast(1);
  }
  // 获取UB内存大小
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  // 获取AiCore的物理核数
  auto coreNum = ascendcPlatform.GetCoreNum();

  // 通过context获取用户传入的数据大小，在本案例中有两个输入，两个输入的维度可能不相同，需要进行广播已达到两个维度相同（潜在问题：有可能存在两个输入不能进行广播，问题先暂存）
  // 广播需要解决两个问题，1、两个输入谁需要广播，2、以什么样的方式广播（总共有两种广播的方式）

  // 1、解决谁需要广播
//   auto inputShap1 = context->GetInputShape(0)->GetStorageShape();
//   auto inputShap2 = context->GetInputShape(1)->GetStorageShape();
  uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  uint32_t inputNum1 = context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  auto dim0 = context->GetInputShape(0)->GetStorageShape().GetDimNum();
  auto dim1 = context->GetInputShape(1)->GetStorageShape().GetDim(2);

  if (inputNum < inputNum1){
      inputNum = inputNum1;
    }

  // 用户输入数据的类型大小（以字节为单位）
  uint32_t inputBytes = 0;
  ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputBytes);

  // 数据的总大小
  uint32_t inputLength = inputNum * inputBytes;

  // 在compute接口中，每次计算需要同时消耗多少个LocalTensor
  uint32_t ubDataNum = 8; // 是根据具体的代码逻辑设定的，不是通过算出来的
  if (dt == 1){
    ubDataNum = 3;
  }

  // aicore每次最多处理多少block块
  uint32_t blockNum = ubSize / 32 / 2 / ubDataNum + 6;

  // 根据每次最多处理的block块，转换成每次最多处理的数据个数
  uint32_t dataNum = blockNum * 32 / inputBytes;

  // 根据32字节对齐计算数据的字节数
  uint32_t inputLengthAlgin32 = (inputLength + 32 - 1) / 32 * 32;

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
//   everyCoreInputBlockNum++;
//   uint32_t bigCoreDataNum = everyCoreInputBlockNum * 32 / inputBytes;

//   uint32_t bigCoreCount = everyCoreInputBlockNum / blockNum;
//   uint32_t bigCoreCarryNum = (everyCoreInputBlockNum % blockNum == 0) ? bigCoreCount : (bigCoreCount + 1);

//   uint32_t bigCoreFinallDealNum = bigCoreDataNum - (dataNum * bigCoreCount);
//   bigCoreFinallDealNum = (bigCoreFinallDealNum == 0) ? dataNum : bigCoreFinallDealNum;

  // 将上述计算的值全部回填到tiling中
  tiling.set_smallCoreDataNum(smallCoreDataNum);
//   tiling.set_bigCoreDataNum(bigCoreDataNum);
  tiling.set_smallCoreCarryNum(smallCoreCarryNum);
//   tiling.set_bigCoreCarryNum(bigCoreCarryNum);
  tiling.set_tileDataNum(dataNum);
  tiling.set_smallCoreFinallDealNum(smallCoreFinallDealNum);
//   tiling.set_bigCoreFinallDealNum(bigCoreFinallDealNum);
//   tiling.set_bigCoreNum(tailBlockNum);
  tiling.set_dataType(dt);
  tiling.set_dim0(dim0);
  tiling.set_dim1(dim1);
//   tiling.set_size1(inputNum);
//   tiling.set_size2(inputNum1);
//   tiling.set_inputShap2(inputShap2);
//   tiling.set_whichSmall(whichSmall);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(coreNum);
  // size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  // currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    gert::Shape* y_shape = context->GetOutputShape(0);
    auto dim0 = x1_shape->GetDimNum();
    auto dim1 = x2_shape->GetDimNum();
    uint32_t size0 = 1;
    uint32_t size1 = 1;
    for (int i = 0; i < dim0; i++){
        size0 *= x1_shape->GetDim(i);
    }
    for (int i = 0; i < dim1; i++){
        size1 *= x2_shape->GetDim(i);
    }
    if (size0 >= size1){
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }else{
        *y_shape = *x2_shape;
        return GRAPH_SUCCESS;
    }
}
}


namespace ops {
class Div : public OpDef {
public:
    explicit Div(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(Div);
}
