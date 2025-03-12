
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LogSumExpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);        // 小核处理的总数据量
TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNumCp);        // 小核处理的总数据量
TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);          // 大核处理的总数据量
TILING_DATA_FIELD_DEF(uint32_t, smallCoreCarryNum);       // 小核搬运数据的次数
TILING_DATA_FIELD_DEF(uint32_t, smallCoreCarryNumCp);       // 小核搬运数据的次数
TILING_DATA_FIELD_DEF(uint32_t, bigCoreCarryNum);         // 大核搬运数据的次数
TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);             // 单核能处理最大数据量
TILING_DATA_FIELD_DEF(uint32_t, smallCoreFinallDealNum);  // 小核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint32_t, smallCoreFinallDealNumCp);  // 小核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint32_t, bigCoreFinallDealNum);    // 大核最后一次处理的数据量
TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);              // 大核个数
TILING_DATA_FIELD_DEF(uint32_t, dataType);                // 运行时数据类型
TILING_DATA_FIELD_DEF(int64_t, dim);
TILING_DATA_FIELD_DEF(bool, keepDim);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSizeCp);
TILING_DATA_FIELD_DEF(uint32_t, ridOfNum);
TILING_DATA_FIELD_DEF(uint32_t, ridOfNumCp);
TILING_DATA_FIELD_DEF(uint32_t, dims);
TILING_DATA_FIELD_DEF(uint32_t, dimSize);
TILING_DATA_FIELD_DEF(uint32_t, dataSize);
TILING_DATA_FIELD_DEF(uint32_t, loop);
TILING_DATA_FIELD_DEF(uint32_t, loopCount);
TILING_DATA_FIELD_DEF(uint32_t, dataSizeCp);
TILING_DATA_FIELD_DEF(uint32_t, loopCp);
TILING_DATA_FIELD_DEF(uint32_t, count);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LogSumExp, LogSumExpTilingData)
}
