
#include "register/tilingdata_base.h"
#include <vector>

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterElementsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);        // 小核处理的总数据量
TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);          // 大核处理的总数据量
TILING_DATA_FIELD_DEF(uint32_t, smallCoreCarryNum);       // 小核搬运数据的次数
TILING_DATA_FIELD_DEF(uint32_t, bigCoreCarryNum);         // 大核搬运数据的次数
TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);             // 单核能处理最大数据量
TILING_DATA_FIELD_DEF(uint32_t, smallCoreFinallDealNum);  // 小核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint32_t, bigCoreFinallDealNum);    // 大核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);              // 大核个数
TILING_DATA_FIELD_DEF(uint32_t, dataType);                // 运行时数据类型
TILING_DATA_FIELD_DEF(uint32_t, reduce);
TILING_DATA_FIELD_DEF(uint32_t, axis);
TILING_DATA_FIELD_DEF(uint32_t, dims);                    // 维度信息
TILING_DATA_FIELD_DEF(uint32_t, interval);                // 数据搬运的间隔
TILING_DATA_FIELD_DEF(uint32_t, ridOfNum);
TILING_DATA_FIELD_DEF(uint32_t, dim0);
TILING_DATA_FIELD_DEF(uint32_t, dim1);
TILING_DATA_FIELD_DEF(uint32_t, dim2);
TILING_DATA_FIELD_DEF(uint32_t, dim3);
TILING_DATA_FIELD_DEF(uint32_t, varDim0);
TILING_DATA_FIELD_DEF(uint32_t, varDim1);
TILING_DATA_FIELD_DEF(uint32_t, varDim2);
TILING_DATA_FIELD_DEF(uint32_t, varDim3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterElements, ScatterElementsTilingData)
}
