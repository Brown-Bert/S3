
#include "log_sum_exp_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  LogSumExpTilingData tiling;
  // 获取dim keep_dim
  auto dimPtr =  context->GetAttrs()->GetListInt(0);
  tiling.set_dimSize(dimPtr->GetSize());
  int64_t dimArray[20];
  for (int i = 0; i < dimPtr->GetSize(); ++i) {
    dimArray[i] = *(dimPtr->GetData() + i);
    // std::cout << "dim = " << dimArray[i] << std::endl;
  }
  auto dim = dimArray[0];
  bool keepDim =  *(context->GetAttrs()->GetBool(1));
 
  // 获取数据类型 0 float 1 half 2 int8 3 int32
  auto dt = context->GetInputTensor(0)->GetDataType();
 //   auto dt1 = context->GetInputTensor(1)->GetDataType();
 
 
  // 获取UB内存大小
  uint64_t ubSize;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
 //   std::cout << "ubsize = " << ubSize << std::endl;
 
  // 获取AiCore的物理核数
  auto coreNum = ascendcPlatform.GetCoreNum();

  // 获取维度信息
  auto dims = context->GetInputShape(0)->GetStorageShape().GetDimNum();

  // 在compute接口中，每次计算需要同时消耗多少个LocalTensor
  uint32_t ubDataNum = 8; // 是根据具体的代码逻辑设定的，不是通过算出来的
 
  // 通过context获取用户传入的数据大小
  uint32_t inputNum = 0;
  uint32_t inputNumCp = 0;
  if (dt == 1 && dims == 3){
    inputNum = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2);
  }else if (dt == 1 && dims == 1){
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
  }else if (dt == 0 && dims == 2){
    inputNum = context->GetInputShape(0)->GetStorageShape().GetDim(1);
  }else if (dt == 0 && dims == 3){
    // ubDataNum = 16;
    inputNum = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2);
  }else{
    // ubDataNum = 16;
    inputNum = context->GetInputShape(0)->GetStorageShape().GetDim(2);
    inputNumCp = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2) * context->GetInputShape(0)->GetStorageShape().GetDim(3);
  }
 
  // 用户输入数据的类型大小（以字节为单位）
  uint32_t inputBytes = 0;
  ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), inputBytes);
 
 
  // 数据的总大小
  uint32_t inputLength = inputNum * inputBytes;
  uint32_t inputLengthCp = inputNumCp * inputBytes;
 
 
  // aicore每次最多处理多少block块
  uint32_t blockNum = ubSize / 32 / 2 / ubDataNum;
 
  // 根据每次最多处理的block块，转换成每次最多处理的数据个数
  uint32_t dataNum = blockNum * 32 / inputBytes;

  uint32_t blockSize = inputNum / dataNum;
  if ((inputNum % dataNum) != 0) {
    blockSize++;
  }
  tiling.set_blockSize(blockSize);
  uint32_t blockSizeCp = inputNumCp / dataNum;
  if ((inputNumCp % dataNum) != 0) {
    blockSizeCp++;
  }
  tiling.set_blockSizeCp(blockSizeCp);
 
  // 根据32字节对齐计算数据的字节数
  uint32_t inputLengthAlgin32 = (inputLength + 32 - 1) / 32 * 32;
  uint32_t inputLengthAlgin32Cp = (inputLengthCp + 32 - 1) / 32 * 32;

  tiling.set_ridOfNum((inputLengthAlgin32 - inputLength) / inputBytes);
  tiling.set_ridOfNumCp((inputLengthAlgin32Cp - inputLengthCp) / inputBytes);
  
 
  // 计算数据需要几个core去执行，如果数据量太小就不需要全部的core去执行
  coreNum = (coreNum < inputLengthAlgin32 / 32) ? coreNum : inputLengthAlgin32 / 32;
  coreNum = (coreNum >= 1) ? coreNum : 1;
 //   std::cout << "coreNum = " << coreNum << std::endl;
 
  // 计算每个core需要处理多少个block
  uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / 32 / coreNum;
  uint32_t everyCoreInputBlockNumCp = inputLengthAlgin32Cp / 32 / coreNum;
 
  // 上面一行不是整除的话，会剩余几个block
  uint32_t tailBlockNum = inputLengthAlgin32 / 32 % coreNum;

 
  /*
      小核
  */
 
  // 计算小核要处理的数据量
  uint32_t smallCoreDataNum = everyCoreInputBlockNum * 32 / inputBytes;
  uint32_t smallCoreDataNumCp = everyCoreInputBlockNumCp * 32 / inputBytes;
 
  // 计算小核，根据每个小核要处理多少block，但是一次只能处理多少个block，计算出需要计算多少次
  uint32_t smallCoreCount = everyCoreInputBlockNum / blockNum;
  uint32_t smallCoreCarryNum = (everyCoreInputBlockNum % blockNum == 0) ? smallCoreCount : (smallCoreCount + 1);

  uint32_t smallCoreCountCp = everyCoreInputBlockNumCp / blockNum;
  uint32_t smallCoreCarryNumCp = (everyCoreInputBlockNumCp % blockNum == 0) ? smallCoreCountCp : (smallCoreCountCp + 1);
 
  // 计算小核最后一次需要处理多少数据
  uint32_t smallCoreFinallDealNum = smallCoreDataNum - (dataNum * smallCoreCount);
  smallCoreFinallDealNum = (smallCoreFinallDealNum == 0) ? dataNum : smallCoreFinallDealNum;

  uint32_t smallCoreFinallDealNumCp = smallCoreDataNumCp - (dataNum * smallCoreCountCp);
  smallCoreFinallDealNumCp = (smallCoreFinallDealNumCp == 0) ? dataNum : smallCoreFinallDealNumCp;

  if (dt == 1 && dims == 3){
    auto loop = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto dataSize = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2);
    tiling.set_loop(loop);
    tiling.set_dataSize(dataSize);
    tiling.set_loopCp(0);
    tiling.set_count(1);
  }else if (dt == 1 && dims == 1){
    tiling.set_loopCp(0);
    tiling.set_loop(1);
  }else if (dt == 0 && dims == 2){
    auto loop = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto dataSize = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    tiling.set_loop(loop);
    tiling.set_dataSize(dataSize);
    tiling.set_loopCp(0);
    tiling.set_count(1);
  }else if (dt == 0 && dims == 3){
    auto loop = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto dataSize = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2);
    tiling.set_loop(loop);
    tiling.set_dataSize(dataSize);
    tiling.set_loopCp(0);
    tiling.set_count(1);
    // dim = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2);
  }else if (dt == 0 && dims == 4){
    auto loopCp = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    auto dataSizeCp = context->GetInputShape(0)->GetStorageShape().GetDim(1) * context->GetInputShape(0)->GetStorageShape().GetDim(2) * context->GetInputShape(0)->GetStorageShape().GetDim(3);
    auto loop = context->GetInputShape(0)->GetStorageShape().GetDim(3);
    auto dataSize = inputLengthAlgin32 / inputBytes;
    auto count = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    tiling.set_loop(loop);
    tiling.set_dataSize(dataSize);
    tiling.set_loopCp(loopCp);
    tiling.set_dataSizeCp(dataSizeCp);
    tiling.set_count(count);
    // dim = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    // auto dimSize = context->GetInputShape(0)->GetOriginShape().GetDim(1);
    // tiling.set_dimSize(dimSize);
    // smallCoreCarryNum = 10;
    // dim = context->GetInputShape(0)->GetStorageShape().GetDim(3);
    // dim = coreNum;
  }
 
 //   std::cout << "smallCoreDataNum = " << smallCoreDataNum << std::endl;
 //   std::cout << "smallCoreCarryNum = " << smallCoreCarryNum << std::endl;
 //   std::cout << "smallCoreFinallDealNum = " << smallCoreFinallDealNum << std::endl;
 //   std::cout << "smallCoreCount = " << smallCoreCount << std::endl;
 
 
 
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
  tiling.set_smallCoreDataNumCp(smallCoreDataNumCp);
  tiling.set_bigCoreDataNum(bigCoreDataNum);
  tiling.set_smallCoreCarryNum(smallCoreCarryNum);
  tiling.set_smallCoreCarryNumCp(smallCoreCarryNumCp);
  tiling.set_bigCoreCarryNum(bigCoreCarryNum);
  tiling.set_tileDataNum(dataNum);
  tiling.set_smallCoreFinallDealNum(smallCoreFinallDealNum);
  tiling.set_smallCoreFinallDealNumCp(smallCoreFinallDealNumCp);
  tiling.set_bigCoreFinallDealNum(bigCoreFinallDealNum);
  tiling.set_bigCoreNum(tailBlockNum);
  tiling.set_dataType(dt);
  tiling.set_dim(dim);
  tiling.set_keepDim(true);
  tiling.set_dims(dims);

 
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  // size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  // currentWorkspace[0] = 0;
  context->SetBlockDim(coreNum);
 
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, x1_shape->GetDim(1));
    y_shape->SetDim(1, x1_shape->GetDim(3));
    // *y_shape = *x1_shape;
    // gert::Shape* z_shape = context->GetOutputShape(1);
    // *z_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class LogSumExp : public OpDef {
public:
    explicit LogSumExp(const char* name) : OpDef(name)
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
        // this->Output("z")
        //     .ParamType(OPTIONAL)
        //     .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
        //     .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        //     .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).ListInt({0});
        this->Attr("keep_dim").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");

    }
};

OP_ADD(LogSumExp);
}
